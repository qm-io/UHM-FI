"""Safe UHM-FI pre-training entry point.

Without ``--execute`` this validates and prints the configuration only.
"""

from uhm_fi.cli import pretrain_main


if __name__ == "__main__":
    pretrain_main()
