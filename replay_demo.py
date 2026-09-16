#!/usr/bin/env python
"""
CrisisState Replay Demo Script.

Executes the replay of the synthetic urban flooding scenario in chronological order,
showing incident creation, claim extraction, explicit contradiction detection,
state snapshot tracking, and attention scoring.
"""

from crisisstate.replay.runner import ReplayRunner


def main():
    runner = ReplayRunner()
    runner.run(verbose=True)


if __name__ == "__main__":
    main()
