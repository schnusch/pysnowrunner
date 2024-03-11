"""
pysnowrunner
Copyright (C) 2024  schnusch

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import os.path
import subprocess
import unittest


class CodeTest(unittest.TestCase):
    directory = os.path.join(os.path.dirname(__file__), "..")
    files = [
        "pysnowrunner",
        "tests",
    ]
    mypy_excludes = [
        "build",
        "vendor",
    ]

    def test_black(self) -> None:
        p = subprocess.run(
            ["black", "--check", "--"] + self.files,
            cwd=self.directory,
        )
        self.assertEqual(p.returncode, 0)

    def test_flake8(self) -> None:
        p = subprocess.run(
            ["flake8", "--"] + self.files,
            cwd=self.directory,
        )
        self.assertEqual(p.returncode, 0)

    def test_isort(self) -> None:
        p = subprocess.run(
            ["isort", "--check", "--profile", "black", "--"] + self.files,
            cwd=self.directory,
        )
        self.assertEqual(p.returncode, 0)

    def test_mypy(self) -> None:
        argv = ["mypy"]
        for exclude in self.mypy_excludes:
            argv.append("--exclude")
            argv.append(exclude)
        p = subprocess.run(argv + ["--", self.directory])
        self.assertEqual(p.returncode, 0)
