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

import dataclasses
import io
import logging
import os
import pprint
import re
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePath, PureWindowsPath
from typing import Any  # noqa: F401
from typing import Set  # noqa: F401
from typing import (
    IO,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from .types import (
    Engine,
    EngineTemplate,
    TemplateDict,
    Truck,
    TruckTire,
    TruckTireTemplate,
    Wheel,
    WheelFriction,
    WheelSoftness,
    WheelTemplate,
)

logger = logging.getLogger(__name__)


@contextmanager
def load_xml_with_error(
    data: bytes,
    pak_path: Optional[PurePath] = None,
    xml_path: Optional[PureWindowsPath] = None,
) -> Iterator[List[ET.Element]]:
    def repl(m: re.Match) -> bytes:
        return m[1] + b"_" + m[2]

    try:
        # expat interprets every \n or \r as a line break
        xdata = data.replace(b"\r\n", b"\n")
        # remove namespace
        xdata = re.sub(rb"(</?region):(default|cis)\b", repl, xdata)
        root = ET.fromstring(b"<root>\n" + xdata + b"</root>")
        yield root.findall("./*")
    except Exception:
        with open("/tmp/error.xml", "wb") as out:
            out.write(b"<!-- ")
            if pak_path is not None:
                out.write(str(pak_path).encode("utf-8"))
                if xml_path is not None:
                    out.write(b": ")
            if xml_path is not None:
                out.write(str(xml_path).encode("utf-8"))
            out.write(b" -->\n")
            out.write(data)
        raise


template_factories = [
    EngineTemplate,
    TruckTireTemplate,
    WheelFriction,
    WheelSoftness,
    WheelTemplate,
]  # type: List[Any]


def load_templates(
    roots: Iterable[ET.Element],
    templates: Optional[TemplateDict] = None,
    assert_include: Optional[str] = None,
) -> TemplateDict:
    """Load <_template> tags."""
    templates = (
        {} if templates is None else dict((k, v.copy()) for k, v in templates.items())
    )
    # templates might not be in order and may derive from a later one
    backlog = []  # type: List[Tuple[Any, ET.Element]]
    for root in roots:
        if root.tag == "_templates":
            assert assert_include is None or root.get("Include") == assert_include
            for cls in template_factories:
                for elem in root.iterfind(f"./{cls.tag}/*"):
                    backlog.append((cls, elem))
    while backlog:
        i = 0
        while i < len(backlog):
            cls, elem = backlog[i]
            assert elem.tag not in templates.get(
                cls, {}
            ), "duplicate template %s/%s" % (cls.tag, elem.tag)
            try:
                templates.setdefault(cls, {})[elem.tag] = cls.from_xml(elem, templates)
            except KeyError:
                i += 1
            else:
                backlog.pop(i)
    return templates


def iter_load_dir_templates(
    pak: zipfile.ZipFile,
    filter: Callable[[PureWindowsPath], bool],
    templates: TemplateDict,
    assert_include: Optional[str] = None,
) -> Iterator[Tuple[PureWindowsPath, List[ET.Element], TemplateDict]]:
    for info in pak.infolist():
        path = PureWindowsPath(info.filename)
        if filter(path):
            with pak.open(info, "r") as fp:
                with load_xml_with_error(fp.read()) as roots:
                    file_templates = load_templates(roots, templates, assert_include)
                    yield (path, roots, file_templates)


def create_dlc_filter(
    prefix: PureWindowsPath,
    infix: PureWindowsPath,
) -> Callable[[PureWindowsPath], bool]:
    def filter(p: PureWindowsPath) -> bool:
        if prefix / infix in p.parents:
            return True
        if prefix / "_dlc" not in p.parents:
            return False
        dlc_base = p.parents[-4]  # [-1] == .
        assert dlc_base.parent == prefix / "_dlc", "%r != %r" % (
            dlc_base.parent,
            prefix / "_dlc",
        )
        return p.is_relative_to(dlc_base / infix)

    return filter


T_DataBase = TypeVar("T_DataBase", bound="DataBase")


@dataclass
class DataBase(object):
    """Data base class not database."""

    @classmethod
    def load(
        cls: Type[T_DataBase],
        initial_pak: Union[str | os.PathLike[str]],
    ) -> T_DataBase:
        """Open *initial_pak* and load game data."""
        pak_path = Path(initial_pak)
        with zipfile.ZipFile(pak_path, "r") as pak:
            return cls._load(pak_path, pak)

    @classmethod
    def _load(
        cls: Type[T_DataBase],
        pak_path: Path,
        pak: zipfile.ZipFile,
    ) -> T_DataBase:
        """This method is called by ``load`` to actually load the game data.
        It is overriden in sub-classes.
        """
        return DataBase()  # type: ignore

    def asdict_non_recursive(self) -> Dict[str, Any]:
        return dict((f.name, getattr(self, f.name)) for f in dataclasses.fields(self))


@dataclass
class StringData(DataBase):
    strings: Dict[str, str]

    logger = globals()["logger"].getChild("StringData")
    string_escape = r'\\[n"\\]'
    string_regex = f'[\\w.,]+|"(?:[^"\\\\]|{string_escape})*"'
    line_regex = re.compile(f"^\\s*({string_regex})\\s+({string_regex})\\s*$")
    str_path = PureWindowsPath("[strings]") / "strings_english.str"

    @staticmethod
    def unescape_string(x: str) -> str:
        def repl(m: re.Match) -> str:
            escapes = {
                "\\n": "\n",
            }
            try:
                return escapes[m[0]]
            except KeyError:
                assert m[0].startswith("\\")
                assert len(m[0]) == 2
                return m[0][1]

        if x.startswith('"'):
            assert x.endswith('"')
            return re.sub(StringData.string_escape, repl, x[1:-1])
        else:
            return x

    @staticmethod
    def load_strings(
        fp: IO[bytes],
        pak_path: Path,
        str_path: PureWindowsPath,
    ) -> Dict[str, str]:
        strings = {}  # type: Dict[str, Tuple[int, str]]
        assert fp.read(2) == b"\xff\xfe"  # BOM
        for lineno, line in enumerate(
            io.TextIOWrapper(fp, encoding="utf-16le"),
            start=1,
        ):
            m = StringData.line_regex.match(line)
            if m is None:
                StringData.logger.error(
                    "%s:%s:%d: cannot parse %r" % (pak_path, str_path, lineno, line)
                )
                continue
            key = StringData.unescape_string(m[1])
            value = StringData.unescape_string(m[2])
            if strings.get(key, (0, value))[1] != value:
                StringData.logger.error(
                    "%s:%s:%d: redefinition of %r as %r, already defined at line %s as %r, overwriting...",
                    pak_path,
                    str_path,
                    lineno,
                    key,
                    value,
                    *strings[key],
                )
            elif key in strings:
                assert strings[key][1] == value
                StringData.logger.warning(
                    "%s:%s:%d: duplicate identical definition of %r as %r, previously at %d",
                    pak_path,
                    str_path,
                    lineno,
                    key,
                    value,
                    strings[key][0],
                )
            strings[key] = (lineno, value)
        return dict((k, v[1]) for k, v in strings.items())

    @classmethod
    def _load(
        cls,
        pak_path: Path,
        pak: zipfile.ZipFile,
    ) -> "StringData":
        base = super()._load(pak_path, pak)
        with pak.open(str(cls.str_path), "r") as fp:
            return StringData(
                strings=cls.load_strings(fp, pak_path, cls.str_path),
                **base.asdict_non_recursive(),
            )


@dataclass
class EngineData(StringData):
    """*engines* is indexed by the basename of the file they were read from
    (without file extension) first, and by the engines *Name* second.
    """

    engines: Dict[str, Dict[str, Engine]]

    @staticmethod
    def load_engines(
        roots: Iterable[ET.Element],
        templates: TemplateDict,
    ) -> Dict[str, Engine]:
        engines = {}  # Dict[str, List[Engine]]
        for root in roots:
            if root.tag == "EngineVariants":
                for child in root.iterfind(f"./{Engine.tag}"):
                    e = Engine.from_xml(child, templates)
                    assert e.Name not in engines
                    engines[e.Name] = e
        return engines

    @classmethod
    def _load(
        cls,
        pak_path: Path,
        pak: zipfile.ZipFile,
    ) -> "EngineData":
        base = super()._load(pak_path, pak)

        all_engines = {}  # type: Dict[str, Dict[str, Engine]]
        for path, roots, file_templates in iter_load_dir_templates(
            pak,
            create_dlc_filter(
                PureWindowsPath("[media]"),
                PureWindowsPath("classes") / "engines",
            ),
            {},  # engines do not rely on truck_templates
        ):
            file_templates[Engine] = {}
            for k, v in file_templates.get(EngineTemplate, {}).items():
                file_templates[Engine][k] = dataclasses.replace(
                    Engine.none(), **v.asdict_non_recursive()
                )

            file_Engine = cls.load_engines(roots, file_templates)
            assert path.stem not in all_engines
            all_engines[path.stem] = file_Engine

        return EngineData(engines=all_engines, **base.asdict_non_recursive())


@dataclass
class TemplateData(EngineData):
    truck_templates: TemplateDict

    @classmethod
    def _load(
        cls,
        pak_path: Path,
        pak: zipfile.ZipFile,
    ) -> "TemplateData":
        base = super()._load(pak_path, pak)
        path = PureWindowsPath("[media]") / "_templates" / "trucks.xml"
        with pak.open(str(path), "r") as fp:
            with load_xml_with_error(fp.read(), pak_path, path) as roots:
                return TemplateData(
                    truck_templates=load_templates(roots),
                    **base.asdict_non_recursive(),
                )


@dataclass
class TireData(TemplateData):
    """*tires* is indexed by the basename of the file they were read from
    (without file extension) first, and by the tire's *Name* second.
    """

    tires: Dict[str, Dict[str, TruckTire]]

    @staticmethod
    def load_tires(
        roots: Iterable[ET.Element],
        templates: TemplateDict,
    ) -> Dict[str, TruckTire]:
        tires = {}  # Dict[str, TruckTire]
        for root in roots:
            if root.tag == "TruckWheels":
                for child in root.iterfind(f"./TruckTires/{TruckTire.tag}"):
                    t = TruckTire.from_xml(child, templates)
                    assert t.Name not in tires
                    tires[t.Name] = t
        return tires

    @classmethod
    def _load(
        cls,
        pak_path: Path,
        pak: zipfile.ZipFile,
    ) -> "TireData":
        base = super()._load(pak_path, pak)

        all_tires = {}  # type: Dict[str, Dict[str, TruckTire]]
        for path, roots, file_templates in iter_load_dir_templates(
            pak,
            create_dlc_filter(
                PureWindowsPath("[media]"),
                PureWindowsPath("classes") / "wheels",
            ),
            base.truck_templates,
            assert_include="trucks",
        ):
            # "convert" ``file_templates[TruckTireTemplate]`` to
            # ``file_templates[TruckTire]`` so the lookup in ``load_tires``
            # works.
            file_templates[TruckTire] = dict(
                (k, dataclasses.replace(TruckTire.none(), **v.asdict_non_recursive()))
                for k, v in file_templates.get(TruckTireTemplate, {}).items()
            )
            file_tires = cls.load_tires(roots, file_templates)
            assert path.stem not in all_tires
            all_tires[path.stem] = file_tires

        return TireData(tires=all_tires, **base.asdict_non_recursive())


@dataclass
class TruckData(TireData):
    trucks: Dict[str, Truck]

    @staticmethod
    def load_trucks(
        roots: Iterable[ET.Element],
        templates: TemplateDict,
        engines: Dict[str, Dict[str, Engine]],
        tires: Dict[str, Dict[str, TruckTire]],
    ) -> Iterator[Truck]:
        for root in roots:
            if root.tag == Truck.tag:
                yield Truck.from_xml(
                    root,
                    templates,
                    engines=engines,
                    tires=tires,
                )

    @classmethod
    def _load(
        cls,
        pak_path: Path,
        pak: zipfile.ZipFile,
    ) -> "TruckData":
        base = super()._load(pak_path, pak)

        all_trucks = {}  # type: Dict[str, Truck]
        for path, roots, file_templates in iter_load_dir_templates(
            pak,
            create_dlc_filter(
                PureWindowsPath("[media]"),
                PureWindowsPath("classes") / "trucks",
            ),
            {},
        ):
            file_templates[Wheel] = dict(
                (k, dataclasses.replace(Wheel.none(), **v.asdict_non_recursive()))
                for k, v in file_templates.get(WheelTemplate, {}).items()
            )
            for truck in cls.load_trucks(
                roots,
                file_templates,
                engines=base.engines,
                tires=base.tires,
            ):
                if path.stem in all_trucks:
                    logger.error(
                        "multiple truck definitions in %s, ignored all but:\n%s",
                        path,
                        pprint.pformat(all_trucks[path.stem]),
                    )
                else:
                    all_trucks[path.stem] = truck

        return cls(trucks=all_trucks, **base.asdict_non_recursive())
