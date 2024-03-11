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
