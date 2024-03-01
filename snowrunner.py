#!/usr/bin/env python3
import dataclasses
import io
import itertools
import json
import logging
import math
import os
import pprint
import re
import uuid
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path, PurePath, PureWindowsPath
from typing import Any, Set  # noqa: F401
from typing import (
    Callable,
    ClassVar,
    Dict,
    IO,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    TypedDict,
    Union,
    cast,
)


logger = logging.getLogger(__name__)

# === GAME DATA TYPES ==================================================

T_Base = TypeVar("T_Base", bound="Base")


class Base(object):
    tag: ClassVar[str]

    @classmethod
    def none(cls: Type[T_Base]) -> T_Base:
        kwargs = dict(
            (f.name, None if f.default is dataclasses.MISSING else f.default)
            for f in dataclasses.fields(cls)  # type: ignore[arg-type]
        )
        return cls(**kwargs)

    @classmethod
    def from_xml(
        cls: Type[T_Base],
        elem: ET.Element,
        templates: "TemplateDict",
    ) -> T_Base:
        template_name = elem.get("_template")
        if template_name is None:
            base = cls.none()
        else:
            try:
                base = templates.get(cls, {})[template_name]
            except KeyError:
                raise KeyError(
                    "cannot find template %r/%s required by %s"
                    % (cls, template_name, ET.tostring(elem, encoding="unicode"))
                )
        return base

    def asdict_non_recursive(self):
        T = TypedDict(
            type(self).__name__,
            dict((f.name, f.type) for f in dataclasses.fields(self)),
        )
        return cast(
            T, dict((f.name, getattr(self, f.name)) for f in dataclasses.fields(self))
        )


T_DecimalAttributes = TypeVar("T_DecimalAttributes", bound="DecimalAttributes")


class DecimalAttributes(Base):
    """``decimal_attributes`` contains names of XML attributes that are
    converted to ``Decimal`` and stored as the object's attributes.
    """

    decimal_attributes: ClassVar[Iterable[str]]

    @classmethod
    def from_xml(
        cls: Type[T_DecimalAttributes],
        elem: ET.Element,
        templates: "TemplateDict",
    ) -> T_DecimalAttributes:
        self = super().from_xml(elem, templates)
        for k in cls.decimal_attributes:
            v = elem.get(k)
            if v is not None:
                self = dataclasses.replace(self, **{k: Decimal(v)})  # type: ignore
        # assert all(
        #     getattr(self, a) is not None
        #     for a in cls.decimal_attributes
        # ), "%r has None attributes: %s" % (self, ET.tostring(elem, encoding="unicode"))
        return self


T_WheelFriction = TypeVar("T_WheelFriction", bound="WheelFriction")


@dataclass
class WheelFriction(DecimalAttributes):
    tag = "WheelFriction"
    decimal_attributes = ("BodyFrictionAsphalt", "BodyFriction", "SubstanceFriction")

    BodyFrictionAsphalt: Decimal
    BodyFriction: Decimal
    SubstanceFriction: Decimal
    IsIgnoreIce: bool = dataclasses.field(default=False, kw_only=True)

    @classmethod
    def from_xml(
        cls: Type[T_WheelFriction],
        elem: ET.Element,
        templates: "TemplateDict",
    ) -> T_WheelFriction:
        self = super().from_xml(elem, templates)
        ignore_ice = elem.get("IsIgnoreIce")
        if ignore_ice is not None and ignore_ice != "false":
            assert ignore_ice == "true"
            self = dataclasses.replace(self, IsIgnoreIce=True)
        return self


@dataclass
class WheelSoftness(DecimalAttributes):
    tag = "WheelSoftness"
    decimal_attributes = ("RadiusOffset", "SoftForceScale")

    RadiusOffset: Decimal
    SoftForceScale: Decimal


T_TruckTireTemplate = TypeVar("T_TruckTireTemplate", bound="TruckTireTemplate")


@dataclass
class TruckTireTemplate(DecimalAttributes):
    tag = "TruckTire"
    decimal_attributes = ("Mass", "RearMassScale")

    Mass: Decimal
    WheelFriction: Optional[WheelFriction]
    WheelSoftness: Optional[WheelSoftness]
    RearMassScale: Decimal = dataclasses.field(default=Decimal(1), kw_only=True)

    @classmethod
    def from_xml(
        cls: Type[T_TruckTireTemplate],
        elem: ET.Element,
        templates: "TemplateDict",
    ) -> T_TruckTireTemplate:
        self = super().from_xml(elem, templates)

        wf = elem.find("./WheelFriction")
        if wf is not None:
            self = dataclasses.replace(
                self, WheelFriction=WheelFriction.from_xml(wf, templates)
            )

        ws = elem.find("./WheelSoftness")
        if ws is not None:
            self = dataclasses.replace(
                self, WheelSoftness=WheelSoftness.from_xml(ws, templates)
            )

        return self


T_TruckTire = TypeVar("T_TruckTire", bound="TruckTire")


@dataclass
class TruckTire(TruckTireTemplate):
    Name: str
    UiName: str
    Price: int
    WheelFriction: WheelFriction

    @classmethod
    def from_xml(
        cls: Type[T_TruckTire],
        elem: ET.Element,
        templates: "TemplateDict",
    ) -> T_TruckTire:
        self = super().from_xml(elem, templates)

        name = elem.get("Name")
        if name is not None:
            self = dataclasses.replace(self, Name=name)

        gamedata = elem.find("./GameData")
        if gamedata is not None:
            uidesc = gamedata.find("./UiDesc")
            if uidesc is not None:
                uiname = uidesc.get("UiName")
                if uiname is not None:
                    self = dataclasses.replace(self, UiName=uiname)

            price = gamedata.get("Price")
            if price is not None:
                self = dataclasses.replace(self, Price=int(price, 10))

        assert (
            self.Name
            is not None
            # and self.UiName is not None
            # and self.Price is not None
        ), "%r has None attributes: %s" % (self, ET.tostring(elem, encoding="unicode"))
        return self


@dataclass
class EngineTemplate(DecimalAttributes):
    tag = "Engine"
    decimal_attributes = (
        "CriticalDamageThreshold",
        "DamageCapacity",
        "EngineResponsiveness",
        "FuelConsumption",
        "DamagedConsumptionModifier",
        "Torque",
        "DamagedMinTorqueMultiplier",
        "DamagedMaxTorqueMultiplier",
        "MaxDeltaAngVel",
    )

    CriticalDamageThreshold: Decimal
    DamageCapacity: Decimal
    EngineResponsiveness: Decimal
    FuelConsumption: Decimal
    Torque: Decimal
    DamagedMinTorqueMultiplier: Decimal
    DamagedMaxTorqueMultiplier: Decimal
    MaxDeltaAngVel: Decimal
    DamagedConsumptionModifier: Decimal = dataclasses.field(
        default=Decimal(1), kw_only=True
    )


T_Engine = TypeVar("T_Engine", bound="Engine")


@dataclass
class Engine(EngineTemplate):
    Name: str
    UiName: str

    @classmethod
    def from_xml(
        cls: Type[T_Engine],
        elem: ET.Element,
        templates: "TemplateDict",
    ) -> T_Engine:
        self = super().from_xml(elem, templates)

        name = elem.get("Name")
        if name is not None:
            self = dataclasses.replace(self, Name=name)

        gamedata = elem.find("./GameData")
        if gamedata is not None:
            uidesc = gamedata.find("./UiDesc")
            if uidesc is not None:
                uiname = uidesc.get("UiName")
                if uiname is not None:
                    self = dataclasses.replace(self, UiName=uiname)

        assert (
            self.Name is not None and self.UiName is not None
        ), "%r has None attributes: %s" % (self, ET.tostring(elem, encoding="unicode"))

        return self


T_CompatibleWheels = TypeVar("T_CompatibleWheels", bound="CompatibleWheels")


@dataclass
class CompatibleWheels(DecimalAttributes):
    tag = "CompatibleWheels"
    decimal_attributes = ("OffsetZ", "Scale")

    Scale: Decimal
    Type: str
    Tires: List[TruckTire]
    OffsetZ: Decimal = dataclasses.field(default=Decimal(0), kw_only=True)

    @classmethod
    def from_xml(  # type: ignore[override]
        cls: Type[T_CompatibleWheels],  # type: ignore[valid-type] # FIXME
        elem: ET.Element,
        templates: "TemplateDict",
        tires: Dict[str, Dict[str, TruckTire]],
    ) -> T_CompatibleWheels:
        self = super().from_xml(elem, templates)  # type: ignore # FIXME
        type = elem.get("Type")
        if type is not None:
            self = dataclasses.replace(self, Type=type)
        assert self.Type is not None, "%r has None attributes: %s" % (
            self,
            ET.tostring(elem, encoding="unicode"),
        )
        self.Tires = list(tires[self.Type].values())
        return self


T_EngineSocket = TypeVar("T_EngineSocket", bound="EngineSocket")


@dataclass
class EngineSocket(Base):
    tag = "EngineSocket"

    Default: str
    Type: List[str]
    Engines: List[Engine]

    @classmethod
    def from_xml(  # type: ignore[override]
        cls: Type[T_EngineSocket],  # type: ignore[valid-type] # FIXME
        elem: ET.Element,
        templates: "TemplateDict",
        engines: Dict[str, Dict[str, Engine]],
    ) -> T_EngineSocket:
        self = super().from_xml(elem, templates)  # type: ignore # FIXME

        default = elem.get("Default")
        if default is not None:
            self = dataclasses.replace(self, Default=default)

        type = elem.get("Type")
        if type is not None:
            self = dataclasses.replace(self, Type=[t.strip() for t in type.split(",")])

        assert (
            self.Default is not None and self.Type is not None
        ), "%r has None attributes: %s" % (self, ET.tostring(elem, encoding="unicode"))

        self.Engines = []
        for t in self.Type:
            self.Engines.extend(engines[t].values())

        return self


T_Truck = TypeVar("T_Truck", bound="Truck")


@dataclass
class Truck(Base):
    tag = "Truck"

    UiName: str
    Price: int
    CompatibleWheels: List[CompatibleWheels]
    EngineSocket: List[EngineSocket]

    @classmethod
    def from_xml(  # type: ignore[override]
        cls: Type[T_Truck],
        elem: ET.Element,
        templates: "TemplateDict",
        engines: Dict[str, Dict[str, Engine]],
        tires: Dict[str, Dict[str, TruckTire]],
    ) -> T_Truck:
        self = super().from_xml(elem, templates)

        self = dataclasses.replace(
            self,
            CompatibleWheels=[
                CompatibleWheels.from_xml(w, templates, tires)
                for w in elem.iterfind("./TruckData/CompatibleWheels")
            ],
            EngineSocket=[
                EngineSocket.from_xml(e, templates, engines)
                for e in elem.iterfind("./TruckData/EngineSocket")
            ],
        )

        gamedata = elem.find("./GameData")
        if gamedata is not None:
            uidesc = gamedata.find("./UiDesc")
            if uidesc is not None:
                # e.g. Ford CLT9000
                region_default = uidesc.find("./region_default")
                if region_default is not None:
                    uidesc = region_default

                uiname = uidesc.get("UiName")
                if uiname is not None:
                    self = dataclasses.replace(self, UiName=uiname)

            price = gamedata.get("Price")
            if price is not None:
                self = dataclasses.replace(self, Price=int(price, 10))

        assert (
            self.UiName is not None and self.CompatibleWheels is not None
        ), "%r has None attributes: %s" % (self, ET.tostring(elem, encoding="unicode"))
        return self


# === GAME DATA LOADERS ================================================


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


TemplateDict = Dict[Type[T_Base], Dict[str, T_Base]]

template_factories = [
    EngineTemplate,
    TruckTireTemplate,
    WheelFriction,
    WheelSoftness,
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
    for root in roots:
        if root.tag == "_templates":
            assert assert_include is None or root.get("Include") == assert_include
            for cls in template_factories:
                for elem in root.iterfind(f"./{cls.tag}/*"):
                    assert elem.tag not in templates.get(
                        cls, {}
                    ), "duplicate template %s/%s" % (cls.tag, elem.tag)
                    templates.setdefault(cls, {})[elem.tag] = cls.from_xml(
                        elem, templates
                    )
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

    def asdict_non_recursive(self):
        T = TypedDict(
            type(self).__name__,
            dict((f.name, f.type) for f in dataclasses.fields(self)),
        )
        return cast(
            T, dict((f.name, getattr(self, f.name)) for f in dataclasses.fields(self))
        )


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
    (without file extension) first, and by the tires *Name* second.
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
    trucks: Dict[str, Dict[str, Truck]]

    @staticmethod
    def load_trucks(
        roots: Iterable[ET.Element],
        templates: TemplateDict,
        engines: Dict[str, Dict[str, Engine]],
        tires: Dict[str, Dict[str, TruckTire]],
    ) -> Dict[str, Truck]:
        trucks = {}  # Dict[str, TruckTire]
        for root in roots:
            if root.tag == Truck.tag:
                t = Truck.from_xml(
                    root,
                    templates,
                    engines=engines,
                    tires=tires,
                )
                assert t.UiName not in trucks
                trucks[t.UiName] = t
        return trucks

    @classmethod
    def _load(
        cls,
        pak_path: Path,
        pak: zipfile.ZipFile,
    ) -> "TruckData":
        base = super()._load(pak_path, pak)

        all_trucks = {}  # type: Dict[str, Dict[str, Truck]]
        for path, roots, file_templates in iter_load_dir_templates(
            pak,
            create_dlc_filter(
                PureWindowsPath("[media]"),
                PureWindowsPath("classes") / "trucks",
            ),
            {},
        ):
            file_trucks = cls.load_trucks(
                roots,
                file_templates,
                engines=base.engines,
                tires=base.tires,
            )
            assert path.stem not in all_trucks
            all_trucks[path.stem] = file_trucks

        return cls(trucks=all_trucks, **base.asdict_non_recursive())


class ChartJsDataset(TypedDict):
    label: str
    data: List[Decimal]


class XMLOutput(object):
    def __init__(self, stream: IO[str], strings: Dict[str, str]):
        self.stream = stream
        self.strings = strings
        self.write_noescape(
            """<script src="https://cdn.jsdelivr.net/npm/chart.js" defer=""></script>
<style>
    .truck {
        display: flex;
        align-items: flex-start;
        gap: .5em;
    }
    .truck > .info {
        flex: 1;
    }
    .tire-charts {
        display: flex;
        overflow-x: scroll;
    }
    .engine-chart, .tire-charts {
        min-height: 12em;
        max-height: 25vh;
    }
    details {
        border-left: .5em solid lightgray;
        padding-left: .5em;
    }
</style>
"""
        )

    @staticmethod
    def escape(x: str) -> str:
        def repl(m: re.Match) -> str:
            return "&#%d;" % ord(m[0])

        return re.sub(r"[^\n !#$%(-;=?-~]", repl, x)

    @staticmethod
    def json_decimal_default(x):
        if isinstance(x, Decimal):
            return float(x)
        else:
            raise TypeError

    def write_noescape(self, x: str) -> None:
        self.stream.write(x)

    def write(self, x: str) -> None:
        self.write_noescape(self.escape(x))

    def draw_chart(
        self,
        options,
        *,
        dom_content_loaded: bool = True,
        klass: Optional[str] = None,
    ) -> None:
        id = str(uuid.uuid4())
        self.write_noescape("<div")
        if klass is not None:
            self.write_noescape(' class="')
            self.write(klass)
            self.write_noescape('"')
        self.write_noescape('><canvas id="')
        self.write(id)
        self.write_noescape('"></canvas><script>\n')
        # for whatever reason HTML does not like it when we escape our script
        self.write_noescape('"use strict";\n')
        if dom_content_loaded:
            self.write_noescape(
                'document.addEventListener("DOMContentLoaded", async () => {\n'
            )
        self.write_noescape(
            "    new Chart(document.getElementById(%s), %s);\n"
            % (
                json.dumps(id),
                json.dumps(options, default=self.json_decimal_default),
            )
        )
        if dom_content_loaded:
            self.write_noescape("});\n")
        self.write_noescape("</script></div>\n")

    def draw_engine_chart(self, engines: Iterable[Engine]) -> None:
        labels = []
        torques = []
        for e in sorted(engines, key=lambda e: e.Torque):
            labels.append(self.strings.get(e.UiName, e.UiName))
            torques.append(e.Torque)
        if not labels:
            return
        self.draw_chart(
            {
                "type": "bar",
                "data": {
                    "labels": labels,
                    "datasets": [
                        ChartJsDataset(
                            label="Torque",
                            data=torques,
                        ),
                    ],
                },
                "options": {
                    "animation": False,
                    "scales": {
                        "x": {
                            "ticks": {
                                "autoSkip": False,
                                "maxRotation": 60,
                            },
                        },
                        "y": {
                            "beginAtZero": True,
                        },
                    },
                },
            },
            klass="engine-chart",
        )

    def draw_tire_charts(self, tires: Iterable[TruckTire]) -> None:
        maximum = Decimal(3)
        grouped_tires = {}  # type: Dict[Tuple[Decimal, Decimal, Decimal, bool], Set[str]]
        for t in tires:
            k = (
                t.WheelFriction.SubstanceFriction,
                t.WheelFriction.BodyFriction,
                t.WheelFriction.BodyFrictionAsphalt,
                t.WheelFriction.IsIgnoreIce,
            )
            grouped_tires.setdefault(k, set()).add(self.strings.get(t.UiName, t.UiName))
            maximum = max(maximum, *k[:3])
        if not grouped_tires:
            return

        series = [
            ChartJsDataset(
                label=", ".join(sorted(names)),
                data=list(data[:3]),
            )
            for data, names in reversed(sorted(grouped_tires.items()))
        ]

        for dataset in series:
            self.draw_chart(
                {
                    "type": "radar",
                    "data": {
                        "labels": ["Mud", "Dirt", "Asphalt"],
                        "datasets": [dataset],
                    },
                    "options": {
                        "animation": False,
                        "scales": {
                            "r": {
                                "min": 0,
                                "max": math.ceil(maximum),
                            },
                        },
                    },
                }
            )
        self.write_noescape("</div>\n")

    def trucks(self, trucks: Dict[str, Dict[str, Truck]]) -> None:
        self.write_noescape("<h1>")
        self.write("Trucks")
        self.write_noescape("</h1>\n")

        for display_name, stem, _, truck in sorted(
            (self.strings[t.UiName], stem, i, t)
            for i, (stem, t) in enumerate(
                itertools.chain.from_iterable(
                    zip(
                        itertools.repeat(stem),
                        ts.values(),
                    )
                    for stem, ts in trucks.items()
                )
            )
        ):
            self.write_noescape("<h2>")
            self.write(display_name)
            self.write_noescape("</h2>\n")
            self.write_noescape('<details open="">\n')
            self.write_noescape('  <div class="truck">\n')
            self.write_noescape('    <img src="')
            self.write("https://raw.githubusercontent.com/VerZsuT/SnowRunner-XML-Editor-Desktop/main/src/images/trucks/%s.jpg" % stem)
            self.write_noescape('">\n')
            self.write_noescape('    <div class="info">\n')
            self.write_noescape("      <details>\n")
            self.write_noescape("        <pre>")
            self.write(pprint.pformat(truck))
            self.write_noescape("</pre>\n")
            self.write_noescape("      </details>\n")
            self.write_noescape("      <h3>Tires</h3>\n")
            self.draw_tire_charts(
                itertools.chain.from_iterable(w.Tires for w in truck.CompatibleWheels),
            )
            self.write_noescape("      <h3>Engines</h3>\n")
            self.draw_engine_chart(
                itertools.chain.from_iterable(es.Engines for es in truck.EngineSocket),
            )
            self.write_noescape("    </div>\n")
            self.write_noescape("  </div>\n")
            self.write_noescape("</details>\n")

    def engines(
        self,
        engines: Dict[str, Dict[str, Engine]],
        trucks: Dict[str, Dict[str, Truck]],
    ) -> None:
        all_engines = []  # type: List[Tuple[str, str, Engine]]
        for stem, e in itertools.chain.from_iterable(
            zip(
                itertools.repeat(stem),
                es.values(),
            )
            for stem, es in engines.items()
        ):
            all_engines.append((self.strings[e.UiName], stem, e))

        truck_engines = {}  # type: Dict[str, List[Truck]]
        for ts in trucks.values():
            for truck in ts.values():
                for socket in truck.EngineSocket:
                    for type in socket.Type:
                        truck_engines.setdefault(type, []).append(truck)

        all_engines = sorted(all_engines, key=lambda e: -e[2].Torque)
        self.write_noescape("<h1>Engines</h1>\n")
        self.draw_engine_chart(e for _, _, e in all_engines)

        for display_name, stem, engine in all_engines:
            self.write_noescape("<h2>")
            self.write(display_name)
            self.write_noescape("</h2>\n")
            self.write_noescape("<p>")
            self.write("Torque: %s" % engine.Torque)
            self.write_noescape("</p>\n")
            engine_trucks = truck_engines.get(stem, [])
            if trucks:
                self.write_noescape("<ul>\n")
                for truck in engine_trucks:
                    self.write_noescape("<li>")
                    self.write(self.strings.get(truck.UiName, truck.UiName))
                    self.write_noescape("</li>\n")
                self.write_noescape("</ul>\n")
            self.write_noescape("<details><pre>")
            self.write(pprint.pformat(engine))
            self.write_noescape("</pre></details>")


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        stream=sys.stderr,
        level=logging.WARNING,
        format="%(name)s %(levelname)-8s :%(lineno)-3d %(message)s",
    )
    # there are lots of "multiple different definitions"
    StringData.logger.setLevel(logging.ERROR + 1)

    data = TruckData.load(
        "/data/sata/steam/SteamLibrary/steamapps/common/SnowRunner/preload/paks/client/initial.pak"
    )
    with open(sys.stdout.fileno(), "w", encoding="ascii", closefd=False) as fp:
        out = XMLOutput(fp, data.strings)
        out.engines(data.engines, data.trucks)
        out.write_noescape("<hr/>\n")
        out.trucks(data.trucks)
