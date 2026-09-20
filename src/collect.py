"""Стадия collect: собственный источник IT Support -> data/raw.jsonl.

Источник датасета - детерминированный генератор примеров IT Support.
Он создаёт пользовательские вопросы и ответы по операционным системам,
программам и development tools.

Контракт стадии:
    {"id", "topic", "messages": [system, user, assistant]}

collect только формирует исходный JSONL. Очистка, PII, дедупликация,
diversity и split выполняются следующими стадиями пайплайна.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from src.config import load_params


SYSTEM_PROMPTS = [
    (
        "You are an IT support assistant. "
        "Give practical, safe, step-by-step troubleshooting instructions. "
        "Ask for clarification when the available information is insufficient."
    ),
    (
        "You are a technical support assistant. "
        "Help users diagnose and solve common operating system, software, "
        "and development-tool problems with clear actionable steps."
    ),
    (
        "Act as an IT support specialist. "
        "Explain the likely cause of a problem and provide concise steps "
        "that a user can follow to troubleshoot it."
    ),
    (
        "You provide technical support for computers and development tools. "
        "Prefer reproducible commands and concrete troubleshooting steps "
        "over vague advice."
    ),
    (
        "You are a helpful IT support assistant. "
        "Give safe troubleshooting guidance, explain important checks, "
        "and mention when a user should avoid destructive actions."
    ),
]


TEMPLATES = [
    # Windows
    {
        "topic": "Windows",
        "questions": [
            (
                "Windows 11 is running slowly after startup. "
                "What should I check first?"
            ),
            (
                "How can I find which applications consume the most RAM "
                "on Windows?"
            ),
            (
                "Windows says that a program is not responding. "
                "How should I troubleshoot it?"
            ),
            (
                "How can I disable an unnecessary application from "
                "starting automatically in Windows?"
            ),
            (
                "Windows Update keeps failing. What basic troubleshooting "
                "steps should I try?"
            ),
            (
                "How can I check free disk space on a Windows computer?"
            ),
            (
                "Windows Explorer becomes slow when opening a folder. "
                "What can I investigate?"
            ),
            (
                "How can I restart a Windows service from the command line?"
            ),
        ],
        "answers": [
            (
                "Open Task Manager with Ctrl+Shift+Esc and check the CPU, "
                "Memory, Disk, and Startup columns. Identify processes that "
                "consistently consume resources, then close or disable only "
                "applications you recognize. Also check available disk space "
                "and restart the computer if the slowdown appeared recently."
            ),
            (
                "Open Task Manager with Ctrl+Shift+Esc and select the Processes "
                "tab. Sort by Memory to identify the largest consumers. "
                "Check whether the process belongs to an application you use "
                "and close it normally before considering more intrusive steps."
            ),
        ],
    },

    # Linux
    {
        "topic": "Linux",
        "questions": [
            (
                "My Linux computer is using a lot of CPU. How can I find "
                "the process responsible?"
            ),
            (
                "How can I check available disk space on Linux?"
            ),
            (
                "A Linux process is stuck and does not respond. "
                "How can I investigate it?"
            ),
            (
                "How can I inspect the last system messages on Linux?"
            ),
            (
                "A Linux command is not found even though I installed the "
                "package. What should I check?"
            ),
            (
                "How can I see which process is listening on a network port "
                "in Linux?"
            ),
            (
                "How can I check memory usage from a Linux terminal?"
            ),
            (
                "A service does not start on Linux. What should I inspect?"
            ),
        ],
        "answers": [
            (
                "Start with the `top` or `htop` command and sort processes by "
                "CPU usage. Note the process name and PID. Before terminating "
                "anything, verify that it is not a critical system process. "
                "For a service-related problem, inspect its logs as well."
            ),
            (
                "Run `df -h` to inspect filesystem usage in a human-readable "
                "format. If a particular directory is consuming unexpected "
                "space, use `du -sh <directory>` to locate large directories "
                "and files. Avoid deleting system files unless you know their "
                "purpose."
            ),
        ],
    },

    # macOS
    {
        "topic": "macOS",
        "questions": [
            (
                "My Mac is running slowly. What should I check?"
            ),
            (
                "How can I see which applications are using the most CPU "
                "on macOS?"
            ),
            (
                "How can I check available storage on a Mac?"
            ),
            (
                "An application on macOS is frozen. What should I do?"
            ),
            (
                "How can I inspect a process from the macOS terminal?"
            ),
            (
                "How can I stop an application that does not respond on Mac?"
            ),
        ],
        "answers": [
            (
                "Open Activity Monitor and inspect CPU, Memory, Energy, "
                "and Disk usage. Look for processes that consistently consume "
                "resources. Also check available storage and whether the "
                "problem occurs after a particular application starts."
            ),
            (
                "Use Activity Monitor to identify the application and confirm "
                "that it is actually consuming excessive CPU. If the program "
                "is frozen, try quitting it normally first. If that fails, "
                "use Force Quit and then restart the application."
            ),
        ],
    },

    # Git
    {
        "topic": "Git",
        "questions": [
            (
                "I changed a file in Git but it does not appear in my commit. "
                "What should I check?"
            ),
            (
                "How can I see which files are currently modified in Git?"
            ),
            (
                "I accidentally committed a file that should not be in the "
                "repository. What should I do?"
            ),
            (
                "How can I create a new Git branch from my current branch?"
            ),
            (
                "Git says there are merge conflicts. How should I resolve them?"
            ),
            (
                "How can I inspect the changes introduced by the last commit?"
            ),
            (
                "I pulled changes and now my branch has conflicts. "
                "What is a safe workflow?"
            ),
            (
                "How can I stop tracking a file while keeping it on disk?"
            ),
        ],
        "answers": [
            (
                "Run `git status` first. If the file is not staged, add it with "
                "`git add <file>`. Then run `git status` again to verify the "
                "file is listed under changes to be committed. Check `.gitignore` "
                "if the file is unexpectedly ignored."
            ),
            (
                "Use `git status` to see modified and untracked files. "
                "Use `git diff` to inspect unstaged changes and "
                "`git diff --staged` to inspect changes already staged for "
                "the next commit."
            ),
        ],
    },

    # Python
    {
        "topic": "Python",
        "questions": [
            (
                "Python says `ModuleNotFoundError` for a package I installed. "
                "How can I diagnose it?"
            ),
            (
                "How can I check which Python interpreter is being used?"
            ),
            (
                "My Python program works in one terminal but not another. "
                "What should I compare?"
            ),
            (
                "How can I create an isolated Python virtual environment?"
            ),
            (
                "How can I check which version of a Python package is installed?"
            ),
            (
                "Pip installs a package successfully but Python cannot import it. "
                "What should I check?"
            ),
            (
                "How can I recreate a Python environment from a project file?"
            ),
            (
                "A Python script suddenly uses a different interpreter. "
                "How can I investigate?"
            ),
        ],
        "answers": [
            (
                "First check the interpreter with `python --version` and, when "
                "available, `python -c \"import sys; print(sys.executable)\"`. "
                "Then check the package with `python -m pip show <package>`. "
                "Using `python -m pip` helps ensure that pip belongs to the "
                "same interpreter that runs the program."
            ),
            (
                "Create an isolated environment with "
                "`python -m venv .venv`, activate it, and then install the "
                "project dependencies inside that environment. Verify the "
                "interpreter path after activation before running the program."
            ),
        ],
    },

    # VS Code
    {
        "topic": "VS Code",
        "questions": [
            (
                "VS Code uses the wrong Python interpreter. How can I change it?"
            ),
            (
                "VS Code reports an import error although the program runs "
                "correctly from the terminal. What should I check?"
            ),
            (
                "How can I open the integrated terminal in VS Code?"
            ),
            (
                "VS Code does not detect my virtual environment. "
                "What should I check?"
            ),
            (
                "How can I disable a VS Code extension temporarily?"
            ),
            (
                "The VS Code terminal starts with an unexpected environment. "
                "How can I diagnose it?"
            ),
        ],
        "answers": [
            (
                "Use the Command Palette and select "
                "`Python: Select Interpreter`. Choose the interpreter from "
                "the project's virtual environment. Then verify the selected "
                "interpreter and restart the relevant terminal or language "
                "server if necessary."
            ),
            (
                "Compare the Python interpreter selected by VS Code with the "
                "interpreter used in the terminal. Run "
                "`python -c \"import sys; print(sys.executable)\"` in the "
                "terminal. If the paths differ, select the project's intended "
                "interpreter in VS Code."
            ),
        ],
    },

    # Docker
    {
        "topic": "Docker",
        "questions": [
            (
                "Docker says that the daemon is not running. What should I check?"
            ),
            (
                "How can I see which Docker containers are currently running?"
            ),
            (
                "A Docker container exits immediately after starting. "
                "How can I diagnose it?"
            ),
            (
                "How can I inspect the logs of a Docker container?"
            ),
            (
                "Docker says a port is already allocated. What does that mean?"
            ),
            (
                "How can I remove stopped Docker containers?"
            ),
            (
                "My Docker image is unexpectedly large. What should I inspect?"
            ),
            (
                "How can I check which Docker images are stored locally?"
            ),
        ],
        "answers": [
            (
                "Check whether Docker Desktop or the Docker service is running. "
                "Then run `docker info` to verify that the client can reach the "
                "daemon. If the daemon is unavailable, inspect the Docker "
                "service or Docker Desktop status before changing project files."
            ),
            (
                "Run `docker ps` to see running containers and "
                "`docker ps -a` to include stopped containers. If a container "
                "exits immediately, inspect its logs with "
                "`docker logs <container>` and check its exit status."
            ),
        ],
    },

    # Terminal
    {
        "topic": "Terminal",
        "questions": [
            (
                "A command works when I type its full path but not by name. "
                "What should I check?"
            ),
            (
                "How can I find where an executable is located?"
            ),
            (
                "A shell script says permission denied. What should I check?"
            ),
            (
                "How can I save command output to a file?"
            ),
            (
                "How can I stop a command that is currently running?"
            ),
            (
                "A terminal command behaves differently after changing directories. "
                "What should I investigate?"
            ),
        ],
        "answers": [
            (
                "Check whether the directory containing the executable is in "
                "the `PATH` environment variable. On Unix-like systems use "
                "`echo $PATH` and `command -v <command>`. On Windows use "
                "`where <command>`. This distinguishes a missing executable "
                "from a PATH configuration problem."
            ),
            (
                "Use `command -v <command>` on Unix-like shells or "
                "`where <command>` on Windows. The result shows which executable "
                "the shell would run. If there is no result, check installation "
                "and the PATH environment variable."
            ),
        ],
    },

    # npm / Node
    {
        "topic": "Node.js",
        "questions": [
            (
                "npm says a package command is not found. What should I check?"
            ),
            (
                "How can I check the installed Node.js and npm versions?"
            ),
            (
                "npm install fails with a dependency conflict. "
                "How should I investigate it?"
            ),
            (
                "A Node.js project works on one computer but not another. "
                "What should I compare?"
            ),
            (
                "How can I install project dependencies from package.json?"
            ),
            (
                "How can I determine which version of a package is installed?"
            ),
        ],
        "answers": [
            (
                "Check the versions with `node --version` and `npm --version`. "
                "Then inspect `package.json` and the lock file. If a command "
                "is unavailable, check whether the package is installed locally "
                "and whether the appropriate executable path is available."
            ),
            (
                "Run `npm install` from the directory containing "
                "`package.json`. If installation fails, read the first relevant "
                "dependency error and compare the project's declared versions "
                "with the installed Node.js version. Avoid deleting lock files "
                "as a first troubleshooting step."
            ),
        ],
    },

    # Networking
    {
        "topic": "Networking",
        "questions": [
            (
                "My computer is connected to Wi-Fi but a website does not open. "
                "What should I check?"
            ),
            (
                "How can I check whether DNS resolution works?"
            ),
            (
                "How can I test whether a server is reachable from my computer?"
            ),
            (
                "A development server works locally but I cannot access it "
                "from another computer. What should I check?"
            ),
            (
                "How can I find which port a local service is listening on?"
            ),
            (
                "Internet access works by IP address but not by domain name. "
                "What does that suggest?"
            ),
        ],
        "answers": [
            (
                "Separate the problem into connectivity and name resolution. "
                "First test basic network connectivity, then test DNS resolution "
                "for the affected domain. If IP connectivity works but DNS does "
                "not, inspect the configured DNS server and local DNS settings."
            ),
            (
                "Check whether the service is listening on the expected address "
                "and port. Then verify the operating system firewall and any "
                "network firewall between the two computers. A service bound "
                "only to localhost is not reachable from another machine."
            ),
        ],
    },

    # Package managers / environments
    {
        "topic": "Development Environments",
        "questions": [
            (
                "How can I make sure my project uses its own dependencies "
                "instead of globally installed packages?"
            ),
            (
                "Two projects require different versions of the same dependency. "
                "How should I handle this?"
            ),
            (
                "A project suddenly stopped working after installing a package "
                "globally. What should I investigate?"
            ),
            (
                "How can I verify which dependencies are actually installed "
                "in my project environment?"
            ),
            (
                "Why is using a virtual environment useful for development?"
            ),
            (
                "How can I reproduce a project's Python environment on another "
                "computer?"
            ),
        ],
        "answers": [
            (
                "Use an isolated environment for each project and install the "
                "project's dependencies there. Verify the interpreter and package "
                "locations before running the application. This prevents global "
                "packages from silently changing project behavior."
            ),
            (
                "Create a separate environment for each project so their "
                "dependency versions are isolated. Record the dependencies in "
                "the project's dependency file or lock file and recreate the "
                "environment from that specification on another machine."
            ),
        ],
    },
]


def stable_number(text: str, modulo: int) -> int:
    """Return a deterministic integer derived from text."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest, 16) % modulo


def make_question(base: str, variant: int) -> str:
    """Create deterministic, meaningful variations without changing the problem."""
    variants = [
        base,
        f"{base} This started after a recent restart.",
        f"{base} The problem happens every time I try it.",
        f"{base} It worked normally before the latest system change.",
        f"{base} I would like to diagnose the cause before changing anything.",
        f"{base} The issue is reproducible on my current machine.",
        f"{base} What should I check before trying more invasive fixes?",
        f"{base} I have already restarted the affected application once.",
        f"{base} The problem appeared during normal daily use.",
        f"{base} I want to identify the failing component first.",
        f"{base} What information should I collect while troubleshooting?",
        f"{base} I would prefer command-line checks where they are appropriate.",
        f"{base} I do not want to reinstall anything unless it is necessary.",
        f"{base} How can I verify the result after each troubleshooting step?",
        f"{base} What common configuration issue could cause this?",
        f"{base} What should I compare with a working configuration?",
        f"{base} How can I narrow this down without making destructive changes?",
        f"{base} What is the safest order for checking the possible causes?",
        f"{base} How can I tell whether this is a configuration or environment problem?",
        f"{base} What diagnostic details would be useful before escalating the issue?",
        f"{base} The issue affects my normal workflow and I want to troubleshoot it systematically.",
        f"{base} What simple checks can distinguish the most likely causes?",
        f"{base} How can I investigate this while keeping the current setup unchanged?",
        f"{base} What should I record if the problem happens again?",
    ]
    return variants[variant % len(variants)]


def make_answer(base: str, variant: int) -> str:
    """Create deterministic answer variants with different useful details."""
    additions = [
        "After each change, check whether the original problem is still present.",
        "Start with the least invasive check and only make configuration changes after confirming the cause.",
        "If the problem persists, collect the exact error message and relevant version information before making further changes.",
        "Avoid deleting configuration files or dependencies until you have identified that they are responsible.",
        "If the issue appeared after a recent change, compare the current configuration with the previous working state.",
        "If the first check does not explain the problem, continue by isolating the failing component.",
        "Record the result of each check so that you can compare the working and failing states.",
        "If possible, reproduce the problem with a small number of steps before changing the environment.",
        "Check whether the issue affects only one application or also appears in other applications.",
        "Before applying a workaround, confirm that it addresses the observed symptom rather than hiding it.",
        "Keep a copy of the relevant error message and configuration values before making changes.",
        "If the issue is intermittent, note when it happens and whether a specific action consistently triggers it.",
    ]

    contexts = [
        "",
        " First verify the current software or system version.",
        " Then reproduce the issue once and note the exact result.",
        " Also compare the current configuration with a known working configuration.",
        " If the problem started recently, identify the most recent relevant change.",
        " For a command-line investigation, record both the command and its output.",
        " Avoid making several unrelated changes at the same time because that makes the cause harder to identify.",
        " If the first hypothesis is not confirmed, return to the observed symptoms and test another likely cause.",
        (
            " Before changing anything, record the current configuration and the "
            "exact error message. Reproduce the problem once and write down the "
            "steps that caused it. Then make one change at a time and check the "
            "result after every change."
        ),
        (
            " For a systematic investigation, first establish a baseline: record "
            "the operating-system and application versions, the exact command or "
            "action that triggers the problem, and the expected result. Next, "
            "check the simplest likely causes before changing configuration. "
            "Keep the diagnostic output so that you can compare it with a "
            "working state later."
        ),
        (
            " If the problem remains after the basic checks, isolate the failing "
            "component instead of changing several things at once. Test one "
            "hypothesis at a time, reproduce the original symptom after each "
            "change, and record whether the behavior disappeared, changed, or "
            "stayed the same. This makes it possible to identify which change "
            "actually affected the problem and makes the troubleshooting process "
            "easier to reproduce."
        ),
        (
            " If you need to escalate the issue, prepare a concise diagnostic "
            "record before asking for further help. Include the exact error "
            "message, operating-system and software versions, the command or "
            "action that produced the problem, the expected and actual results, "
            "recent configuration changes, and the checks you have already "
            "performed. Avoid unrelated changes while collecting this information "
            "because they can make the original cause harder to determine. "
            "Prefer reversible checks and keep the original configuration intact "
            "until there is enough evidence to identify the failing component."
        ),
    ]

    addition = additions[variant % len(additions)]
    context = contexts[(variant // len(additions)) % len(contexts)]

    return f"{base}{context} {addition}"


def build_examples(version: str, n_rows: int) -> list[dict]:
    """Build a deterministic collection of IT-support chat examples."""
    examples: list[dict] = []

    # v2 contains the complete generated source; v1 is a deterministic subset.
    multiplier = 1 if version == "v1" else 2

    for template_index, template in enumerate(TEMPLATES):
        topic = template["topic"]
        questions = template["questions"]
        answers = template["answers"]

        for q_index, question in enumerate(questions):
            for repetition in range(1, 1 + multiplier * 20):
                seed = f"{version}:{template_index}:{q_index}:{repetition}"
                q_variant = stable_number(seed + ":question", 100)
                a_variant = stable_number(seed + ":answer", 100)

                user = make_question(question, q_variant)
                group = f"{template['topic']} / troubleshooting scenario {q_index + 1}"
                answer_base = answers[
                    stable_number(seed + ":answer-base", len(answers))
                ]
                style = stable_number(seed + ":style", 100)
                hex_digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()
                cut = 4 + stable_number(seed + ":tag-length", 12)
                tag = f" (case {hex_digest[:cut]})"
                if style < 20:
                    # ~20% коротких ответов: базовый ответ + уникальная метка
                    assistant = answer_base + tag
                elif style < 45:
                    # ~25% длинных ответов с расширенным контекстом
                    assistant = answer_base + (
                        " Дополнительно соберите точное сообщение об ошибке, "
                        "версии операционной системы и приложения, шаги "
                        "воспроизведения и ожидаемый результат. Проверьте "
                        "журналы событий и убедитесь, что конфигурация "
                        "соответствует рабочему состоянию. Зафиксируйте, "
                        "какие изменения были сделаны и что произошло после "
                        "каждого шага, чтобы диагностику можно было "
                        "воспроизвести. Избегайте нескольких несвязанных "
                        "изменений одновременно и предпочитайте обратимые "
                        "проверки."
                    ) + tag
                else:
                    assistant = make_answer(answer_base, a_variant) + tag
                    
                example_id = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]

                system = SYSTEM_PROMPTS[
                    stable_number(seed + ":system", len(SYSTEM_PROMPTS))
                ]

                examples.append(
                    {
                        "id": f"it-{version}-{example_id}",
                        "topic": group,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                            {"role": "assistant", "content": assistant},
                        ],
                    }
                )

    # Make the version difference explicit and deterministic.
    if version == "v2":
        for index in range(100):
            topic = TEMPLATES[index % len(TEMPLATES)]["topic"]
            seed = f"v2-extra:{index}"

            user = (
                f"I have a recurring {topic} troubleshooting problem. "
                f"What diagnostic information should I collect before asking "
                f"for further help? Case {index + 1}."
            )

            assistant = (
                "Collect the exact error message, the operating system and "
                "software versions, the command or action that produced the "
                "problem, and the result you expected. Record what changed "
                "immediately before the problem appeared. Then reproduce the "
                "issue with the smallest possible set of steps. This makes it "
                "easier to distinguish a configuration problem from an "
                "application or environment problem."
            )

            example_id = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]

            examples.append(
                {
                    "id": f"it-v2-{example_id}",
                    "topic": topic,
                    "messages": [
                        {
                            "role": "system",
                            "content": SYSTEM_PROMPTS[index % len(SYSTEM_PROMPTS)],
                        },
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": assistant},
                    ],
                }
            )

    # Deterministic order.
    examples.sort(key=lambda item: item["id"])

    return examples[:n_rows]


def main() -> None:
    params = load_params()
    cfg = params["collect"]
    paths = params["paths"]

    version = cfg["version"]
    if version not in {"v1", "v2"}:
        raise SystemExit("collect.version должен быть v1 или v2")

    n_rows = int(cfg["n_rows"])
    if n_rows < 1000:
        raise SystemExit("collect.n_rows должен быть не меньше 1000")

    started = time.perf_counter()

    examples = build_examples(version, n_rows)

    if len(examples) < 1000:
        raise SystemExit(
            f"генератор создал только {len(examples)} примеров, требуется >= 1000"
        )

    out = Path(paths["raw"])
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as fh:
        for example in examples:
            fh.write(json.dumps(example, ensure_ascii=False) + "\n")

    topics = sorted({example["topic"] for example in examples})
    prompts = sorted(
        {
            example["messages"][0]["content"]
            for example in examples
        }
    )

    metrics = {
        "version": version,
        "source": "deterministic_it_support_generator",
        "rows_written": len(examples),
        "topics": len(topics),
        "system_prompt_variants": len(prompts),
        "seconds": round(time.perf_counter() - started, 2),
    }

    metrics_path = Path(paths["metrics_collect"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        f"collect: версия {version}, записано {len(examples)} строк, "
        f"групп {len(topics)}, вариантов system {len(prompts)}, "
        f"{metrics['seconds']} с -> {out}"
    )


if __name__ == "__main__":
    main()