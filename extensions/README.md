# Extension tasks

Each subdirectory holds one contributed paradigm in `task.py`. These tasks sit outside the 24-task paper suite. `python -m s1cogbench eval --tasks extensions` runs all of them, and a task ID runs one. `--tasks all` still means the 27 paper modes.

Extension results are compared only within the same task and protocol. They never enter the paper tables, and their leaderboards form a separate group on the project page. A protocol change takes a new `protocol` identifier, so earlier results stay labelled with the version they used.

`_template/` is a starting point and is not loaded. See [docs/ADD_TASK.md](../docs/ADD_TASK.md) for how to write, check and submit a task.
