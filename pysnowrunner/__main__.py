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

import logging
import pprint
import sys

from .loader import StringData, TruckData

logging.basicConfig(
    stream=sys.stderr,
    level=logging.WARNING,
    format="%(name)s %(levelname)-8s :%(lineno)-3d %(message)s",
)
# there are lots of "multiple different definitions"
StringData.logger.setLevel(logging.ERROR + 1)

data = TruckData.load("initial.pak")
pprint.pprint(data)
