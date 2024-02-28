import dataclasses
import io
import itertools
import json
import logging
import os
import pathlib
import pprint
import re
import uuid
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal
from typing import (
    Callable,
    ClassVar,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    TypedDict,
    Union,
)


logger = logging.getLogger(__name__)


'''
def iter_pak_files(
    root: str,
    ignore_error: bool = False,
) -> Iterator[Tuple[pathlib.Path, zipfile.ZipFile]]:
    for root, _, files, dir_fd in os.fwalk(root):
        path = pathlib.Path(root)
        for name in files:
            if name.lower().endswith(".pak"):
                fd = os.open(name, os.O_RDONLY, dir_fd=dir_fd)
                try:
                    with open(fd, "rb", closefd=False) as fp:
                        try:
                            with zipfile.ZipFile(fp, "r") as pak:
                                yield (path / name, pak)
                        except zipfile.BadZipFile:
                            if not ignore_error:
                                raise
                finally:
                    os.close(fd)


class Strings(object):
    logger = globals()["logger"].getChild("Strings")  # type: logging.Logger

    string_escape = r'\\[n"\\]'  # FIXME
    string_regex = f'[\\w.,]+|"(?:[^"\\\\]|{string_escape})*"'
    line_regex = re.compile(f"^\\s*({string_regex})\\s+({string_regex})\\s*$")

    def __init__(self, root: str):
        self.strings = (
            {}
        )  # type: Dict[str, Dict[Tuple[pathlib.Path, pathlib.PureWindowsPath], str]]
        self.language = (
            None,
            pathlib.PureWindowsPath("[strings]") / "strings_english.str",
        )  # type: Tuple[Optional[pathlib.Path], pathlib.PureWindowsPath]
        self.load(root)

    @classmethod
    def unescape_string(cls, x: str) -> str:
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
            return re.sub(cls.string_escape, repl, x[1:-1])
        else:
            return x

    def load(self, root: str) -> None:
        for pak_path, pak in iter_pak_files(root):
            for info in pak.infolist():
                str_path = pathlib.PureWindowsPath(info.filename)
                if str_path.suffix.lower() == ".str":
                    with pak.open(info, "r") as fp:
                        assert fp.read(2) == b"\xff\xfe"  # BOM
                        for lineno, line in enumerate(
                            io.TextIOWrapper(fp, encoding="utf-16-le"),
                            start=1,
                        ):
                            m = self.line_regex.match(line)
                            if m is None:
                                self.logger.error(
                                    "cannot parse line %r in %s: %s at line %d"
                                    % (line, str(pak_path), str(str_path), lineno)
                                )
                                continue
                            key = self.unescape_string(m[1])
                            value = self.unescape_string(m[2])
                            variants = self.strings.setdefault(key, {})
                            variant = (pak_path, str_path)
                            if variants.get(variant, value) != value:
                                self.logger.error(
                                    "multiple different definitions of %r in %s: %s: %r, %r; overwriting",
                                    key,
                                    *variant,
                                    variants[variant],
                                    value,
                                )
                            elif variant in variants:
                                assert variants[variant] == value
                                self.logger.info(
                                    "duplicate string definition %r in %s: %s",
                                    key,
                                    *variant,
                                )
                            variants[variant] = value

    @property
    def languages(self) -> Set[Tuple[pathlib.Path, pathlib.PureWindowsPath]]:
        languages = set()
        for key, variants in self.strings.items():
            for variant in variants:
                languages.add(variant)
        return languages

    def __getitem__(self, key: str) -> str:
        variants = self.strings[key]
        for (pak_path, str_path), value in variants.items():
            if (
                self.language[0] is None or pak_path == self.language[0]
            ) and str_path == self.language[1]:
                return value
        try:
            return next(iter(variants.values()))
        except StopIteration:
            raise KeyError

    def __iter__(self) -> Iterator[str]:
        return iter(self.strings)


class FallbackHTMLParser(html.parser.HTMLParser):
    """``html.parser.Parser`` handles duplicate attributes."""
    logger = globals()["logger"].getChild("FallbackHTMLParser")  # type: logging.Logger

    def __init__(self, pak: pathlib.PurePath, name: pathlib.PureWindowsPath) -> None:
        super().__init__()
        self.pak = pak
        self.name = name
        self.stack = [
            (ET.Element("root"), "\n"),
        ]  # type: List[Tuple[ET.Element, str]]
        self.prev = None  # type: Optional[ET.Element]

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, str]]) -> None:
        elem = ET.Element(tag, dict(attrs), text="")
        for i, (a, av) in enumerate(attrs):
            double = [bv for b, bv in attrs[i + 1 :] if b == a]
            if double:
                self.logger.warning(
                    "duplicate attribute %s, ignored %r, kept %r",
                    a,
                    [av] + double[:-1],
                    double[-1],
                )
                assert elem.get(a) == double[-1], repr(elem.get(k))
        parent, indent = self.stack[-1]
        parent.append(elem)
        if self.prev is None:
            parent.text = indent
        else:
            self.prev.tail = indent
            self.prev = None
        self.stack.append((elem, indent + "  "))

    def handle_startendtag(self, tag: str, attrs: List[Tuple[str, str]]) -> None:
        if (
            tag == "cloud"
            and not attrs
            and self.name == pathlib.PureWindowsPath("[media]\\classes\\skies\\sky_us_01_ttt.xml")
        ):
            # There is a bug in initial.pak: [media]/classes/skies/sky_us_01_ttt.xml
            # with a misplaced <Cloud/> instead of </Cloud>.
            self.logger.warning(
                "handling misplaced <%s/> in %s: %s",
                tag,
                self.pak,
                self.name,
            )
            self.handle_endtag(tag)
        else:
            super().handle_startendtag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        popped, _ = self.stack.pop()
        assert tag == popped.tag, "%r != %r\n%s" % (tag, popped.tag, ET.tostring(self.stack[0][0], encoding="unicode"))
        if self.prev is not None:
            _, indent = self.stack[-1]
            self.prev.tail = indent
        self.prev = popped


def parse_html(
    data: bytes,
    pak: pathlib.PurePath,
    name: pathlib.PureWindowsPath,
) -> List[ET.Element]:
    parser = FallbackHTMLParser(pak, name)
    try:
        parser.feed(data.decode("utf-8"))
    finally:
        parser.close()
    assert len(parser.stack) == 1, (
        "unexpected element stack:\n"
        + "\n".join(ET.tostring(e, encoding="unicode") for e, _ in parser.stack),
    )
    return parser.stack[0][0].findall("./*")


def lowercase_element_recursive(elem: ET.Element) -> ET.Element:
    new = ET.Element(
        elem.tag.lower(),
        attrib=dict((k.lower(), v) for k, v in elem.items()),
    )
    if elem.text is not None:
        new.text = elem.text
    if elem.tail is not None:
        new.tail = elem.tail
    for c in elem.iterfind("./*"):
        new.append(lowercase_element_recursive(c))
    return new


def parse_xml(data: bytes, pak: str, name: str) -> List[ET.Element]:
    try:
        root = ET.fromstring(b"<root>\n" + data + b"</root>")
        return [lowercase_element_recursive(e) for e in root.iterfind("./*")]
    except ET.ParseError:
        return parse_html(b"\n" + data, pak, name)


def iter_xml_data(root: str) -> Iterator[
    Tuple[
        List[ET.Element],
        pathlib.Path,
        pathlib.PureWindowsPath,
    ]
]:
    for pak_path, pak in iter_pak_files(root):
        for info in pak.infolist():
            xml_path = pathlib.PureWindowsPath(info.filename)
            if xml_path.suffix.lower() == ".xml":
                with pak.open(info, "r") as fp:
                    data = fp.read().replace(b"\r\n", b"\n")
                    try:
                        elems = parse_xml(data, pak_path, xml_path)
                    except:
                        with open("/tmp/error.xml", "wb") as fp2:
                            fp2.write(b"<!-- ")
                            fp2.write(str(pak_path).encode("utf-8"))
                            fp2.write(b": ")
                            fp2.write(str(xml_path).encode("utf-8"))
                            fp2.write(b" -->\n")
                            fp2.write(data)
                        raise
                    else:
                        yield (elems, pak_path, xml_path)
'''

T = TypeVar("T", bound="Base")


class Base(object):
    tag: ClassVar[str]

    @classmethod
    def none(cls: Type[T]) -> T:
        kwargs = dict(
            (f.name, None if f.default is dataclasses.MISSING else f.default)
            for f in dataclasses.fields(cls)
        )
        return cls(**kwargs)

    @classmethod
    def from_xml(
        cls: Type[T],
        elem: ET.Element,
        templates: "TemplateDict",
    ) -> T:
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


class DecimalAttributes(Base):
    decimal_attributes: ClassVar[Iterable[str]]

    @classmethod
    def from_xml(
        cls: Type[T],
        elem: ET.Element,
        templates: "TemplateDict",
    ) -> T:
        base = super().from_xml(elem, templates)
        new = {}
        for attr in cls.decimal_attributes:
            new[attr] = elem.get(attr)
        replace = dict((k, Decimal(v)) for k, v in new.items() if v is not None)
        self = dataclasses.replace(base, **replace)
        # assert all(
        #     getattr(self, a) is not None
        #     for a in cls.decimal_attributes
        # ), "%r has None attributes: %s" % (self, ET.tostring(elem, encoding="unicode"))
        return self


@dataclass
class WheelFriction(DecimalAttributes):
    tag = "WheelFriction"
    decimal_attributes = ("BodyFrictionAsphalt", "BodyFriction", "SubstanceFriction")

    BodyFrictionAsphalt: Decimal
    BodyFriction: Decimal
    SubstanceFriction: Decimal


@dataclass
class WheelSoftness(DecimalAttributes):
    tag = "WheelSoftness"
    decimal_attributes = ("RadiusOffset", "SoftForceScale")

    RadiusOffset: Decimal
    SoftForceScale: Decimal


@dataclass
class TruckTireTemplate(DecimalAttributes):
    tag = "TruckTire"
    decimal_attributes = ("Mass", "RearMassScale")

    Mass: Decimal
    RearMassScale: Decimal = dataclasses.field(default=Decimal(1), kw_only=True)
    WheelFriction: Optional[WheelFriction]
    WheelSoftness: Optional[WheelSoftness]

    @classmethod
    def from_xml(
        cls: Type[T],
        elem: ET.Element,
        templates: "TemplateDict",
    ) -> T:
        base = super().from_xml(elem, templates)

        replace = {}

        wf = elem.find("./WheelFriction")
        if wf is not None:
            replace["WheelFriction"] = WheelFriction.from_xml(wf, templates)

        ws = elem.find("./WheelSoftness")
        if ws is not None:
            replace["WheelSoftness"] = WheelSoftness.from_xml(ws, templates)

        return dataclasses.replace(base, **replace)


@dataclass
class TruckTire(TruckTireTemplate):
    Name: str
    UiName: str
    Price: int

    @classmethod
    def from_xml(
        cls: Type[T],
        elem: ET.Element,
        templates: "TemplateDict",
    ) -> T:
        base = super().from_xml(elem, templates)

        replace = {}

        name = elem.get("Name")
        if name is not None:
            replace["Name"] = name

        gamedata = elem.find("./GameData")
        if gamedata is not None:
            uidesc = gamedata.find("./UiDesc")
            if uidesc is not None:
                uiname = uidesc.get("UiName")
                if uiname is not None:
                    replace["UiName"] = uiname

            price = gamedata.get("Price")
            if price is not None:
                replace["Price"] = int(price, 10)

        self = dataclasses.replace(base, **replace)
        # assert (
        #     self.Name is not None
        #     # and self.UiName is not None
        #     # and self.Price is not None
        # ), "%r has None attributes: %s" % (self, ET.tostring(elem, encoding="unicode"))
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
    DamagedConsumptionModifier: Decimal = dataclasses.field(default=Decimal(1), kw_only=True)
    Torque: Decimal
    DamagedMinTorqueMultiplier: Decimal
    DamagedMaxTorqueMultiplier: Decimal
    MaxDeltaAngVel: Decimal


@dataclass
class Engine(EngineTemplate):
    Name: str
    UiName: str

    @classmethod
    def from_xml(
        cls: Type[T],
        elem: ET.Element,
        templates: "TemplateDict",
    ) -> T:
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
            self.Name is not None
            and self.UiName is not None
        ), "%r has None attributes: %s" % (self, ET.tostring(elem, encoding="unicode"))

        return self


@dataclass
class CompatibleWheels(DecimalAttributes):
    tag = "CompatibleWheels"
    decimal_attributes = ("OffsetZ", "Scale")

    OffsetZ: Decimal = dataclasses.field(default=Decimal(0), kw_only=True)
    Scale: Decimal
    Type: str
    Tires: List[TruckTire]

    @classmethod
    def from_xml(
        cls: Type[T],
        elem: ET.Element,
        templates: "TemplateDict",
        tires: Dict[str, Dict[str, TruckTire]],
    ) -> T:
        self = super().from_xml(elem, templates)
        type = elem.get("Type")
        if type is not None:
            self = dataclasses.replace(self, Type=type)
        assert self.Type is not None, "%r has None attributes: %s" % (self, ET.tostring(elem, encoding="unicode"))
        self.Tires = list(tires[self.Type].values())
        return self


@dataclass
class EngineSocket(Base):
    tag = "EngineSocket"

    Default: str
    Type: str
    Engines: List[Engine]

    @classmethod
    def from_xml(
        cls: Type[T],
        elem: ET.Element,
        templates: "TemplateDict",
        engines: Dict[str, Dict[str, Engine]],
    ) -> T:
        self = super().from_xml(elem, templates)

        default = elem.get("Default")
        if default is not None:
            self = dataclasses.replace(self, Default=default)

        type = elem.get("Type")
        if type is not None:
            self = dataclasses.replace(self, Type=type)

        assert (
            self.Default is not None
            and self.Type is not None
        ), "%r has None attributes: %s" % (self, ET.tostring(elem, encoding="unicode"))

        self.Engines = []
        for t in self.Type.split(","):
            self.Engines.extend(engines[t.strip()].values())

        return self


@dataclass
class Truck(Base):
    tag = "Truck"

    UiName: str
    Price: int
    CompatibleWheels: List[str]
    EngineSocket: List[EngineSocket]

    @classmethod
    def from_xml(
        cls: Type[T],
        elem: ET.Element,
        templates: "TemplateDict",
        engines: Dict[str, Dict[str, Engine]],
        tires: Dict[str, Dict[str, TruckTire]],
    ) -> T:
        base = super().from_xml(elem, templates)

        replace = {
            "CompatibleWheels": [
                CompatibleWheels.from_xml(w, templates, tires)
                for w in elem.iterfind("./TruckData/CompatibleWheels")
            ],
            "EngineSocket": [
                EngineSocket.from_xml(e, templates, engines)
                for e in elem.iterfind("./TruckData/EngineSocket")
            ],
        }

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
                    replace["UiName"] = uiname

            price = gamedata.get("Price")
            if price is not None:
                replace["Price"] = int(price, 10)

        self = dataclasses.replace(base, **replace)
        assert (
            self.UiName is not None
            and self.CompatibleWheels is not None
        ), "%r has None attributes: %s" % (self, ET.tostring(elem, encoding="unicode"))
        return self


@contextmanager
def load_xml_with_error(
    data: str,
    pak_path: Optional[pathlib.PurePath] = None,
    xml_path: Optional[pathlib.PureWindowsPath] = None,
) -> Iterator[List[ET.Element]]:
    def repl(m: re.Match) -> bytes:
        return m[1] + b"_" + m[2]

    try:
        # expat interprets every \n or \r as a line break
        xdata = data.replace(b"\r\n", b"\n")
        # remove 
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


T = TypeVar("T", bound=Base)
TemplateDict = Dict[Type[T], Dict[str, T]]

template_factories = [
    EngineTemplate,
    TruckTireTemplate,
    WheelFriction,
    WheelSoftness,
]  # type: List[Type[T]]


def load_templates(
    roots: Iterable[ET.Element],
    templates: Optional[TemplateDict] = None,
    assert_include: Optional[str] = None,
) -> TemplateDict:
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
                    templates.setdefault(cls, {})[elem.tag] = cls.from_xml(elem, templates)
    return templates


def iter_load_templates_dir(
    pak: zipfile.ZipFile,
    filter: Callable[[pathlib.PureWindowsPath], bool],
    templates: TemplateDict,
    assert_include: Optional[str] = None,
) -> Iterator[Tuple[pathlib.PureWindowsPath, bytes, TemplateDict]]:
    for info in pak.infolist():
        path = pathlib.PureWindowsPath(info.filename)
        if filter(path):
            with pak.open(info, "r") as fp:
                with load_xml_with_error(fp.read()) as roots:
                    file_templates = load_templates(roots, templates, assert_include)
                    yield (path, roots, file_templates)


def create_filter(
    prefix: pathlib.PureWindowsPath,
    infix: pathlib.PureWindowsPath,
) -> Callable[[pathlib.PureWindowsPath], bool]:
    def filter(p: pathlib.PureWindowsPath) -> bool:
        if prefix / infix in p.parents:
            return True
        if prefix / "_dlc" not in p.parents:
            return False
        dlc_base = p.parents[-4]  # [-1] == .
        assert dlc_base.parent == prefix / "_dlc", (
            "%r != %r"
            % (dlc_base.parent, prefix / "_dlc")
        )
        return p.is_relative_to(dlc_base / infix)

    return filter


def asdict_non_recursive(x):
    return dict(
        (f.name, getattr(x, f.name)) for f in dataclasses.fields(x)
    )


def load_engines(
    roots: Iterable[ET.Element],
    templates: TemplateDict,
) -> Dict[str, TruckTire]:
    engines = {}  # Dict[str, List[Engine]]
    for root in roots:
        if root.tag == "EngineVariants":
            for child in root.iterfind(f"./{Engine.tag}"):
                e = Engine.from_xml(child, templates)
                assert e.Name not in engines
                engines[e.Name] = e
    return engines


def load_all_engines(
    pak: zipfile.ZipFile,
    templates: TemplateDict,
) -> Dict[str, Dict[str, TruckTire]]:
    all_engines = {}  # type: Dict[str, List[TruckTire]]
    for path, roots, file_templates in iter_load_templates_dir(
        pak,
        create_filter(
            pathlib.PureWindowsPath("[media]"),
            pathlib.PureWindowsPath("classes") / "engines",
        ),
        templates,
    ):
        file_templates[Engine] = dict(
            (k, dataclasses.replace(Engine.none(), **asdict_non_recursive(v)))
            for k, v in file_templates.get(EngineTemplate, {}).items()
        )
        file_Engine = load_engines(roots, file_templates)
        assert path.stem not in all_engines
        all_engines[path.stem] = file_Engine
    return all_engines


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


def load_all_tires(
    pak: zipfile.ZipFile,
    templates: TemplateDict,
) -> Dict[str, Dict[str, TruckTire]]:
    all_tires = {}  # type: Dict[str, List[TruckTire]]
    for path, roots, file_templates in iter_load_templates_dir(
        pak,
        create_filter(
            pathlib.PureWindowsPath("[media]"),
            pathlib.PureWindowsPath("classes") / "wheels",
        ),
        templates,
        assert_include="trucks",
    ):
        file_templates[TruckTire] = dict(
            (k, dataclasses.replace(TruckTire.none(), **asdict_non_recursive(v)))
            for k, v in file_templates.get(TruckTireTemplate, {}).items()
        )
        file_tires = load_tires(roots, file_templates)
        assert path.stem not in all_tires
        all_tires[path.stem] = file_tires
    return all_tires


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


def load_all_trucks(
    pak: zipfile.ZipFile,
    templates: TemplateDict,
    engines: Dict[str, Dict[str, Engine]],
    tires: Dict[str, Dict[str, TruckTire]],
) -> Dict[str, List[TruckTire]]:
    all_trucks = {}  # type: Dict[str, List[TruckTire]]
    for _, roots, file_templates in iter_load_templates_dir(
        pak,
        create_filter(
            pathlib.PureWindowsPath("[media]"),
            pathlib.PureWindowsPath("classes") / "trucks",
        ),
        templates,
    ):
        file_trucks = load_trucks(
            roots,
            file_templates,
            engines=engines,
            tires=tires,
        )
        for k, v in file_trucks.items():
            all_trucks.setdefault(k, []).append(v)
    return all_trucks


class Strings(object):
    logger = globals()["logger"].getChild("Strings")  # type: logging.Logger

    string_escape = r'\\[n"\\]'
    string_regex = f'[\\w.,]+|"(?:[^"\\\\]|{string_escape})*"'
    line_regex = re.compile(f"^\\s*({string_regex})\\s+({string_regex})\\s*$")

    def __init__(self, root: str):
        self.strings = (
            {}
        )  # type: Dict[str, Dict[Tuple[pathlib.Path, pathlib.PureWindowsPath], str]]
        self.language = (
            None,
            pathlib.PureWindowsPath("[strings]") / "strings_english.str",
        )  # type: Tuple[Optional[pathlib.Path], pathlib.PureWindowsPath]
        self.load(root)

    @classmethod
    def unescape_string(cls, x: str) -> str:
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
            return re.sub(cls.string_escape, repl, x[1:-1])
        else:
            return x

    @classmethod
    def load(cls, pak: zipfile.ZipFile, path: pathlib.PureWindowsPath) -> Dict[str, str]:
        strings = {}
        with pak.open(str(path), "r") as fp:
            assert fp.read(2) == b"\xff\xfe"  # BOM
            for lineno, line in enumerate(
                io.TextIOWrapper(fp, encoding="utf-16le"),
                start=1,
            ):
                m = cls.line_regex.match(line)
                if m is None:
                    cls.logger.error(
                        "cannot parse line %r in %s at line %d"
                        % (line, path, lineno)
                    )
                    continue
                key = cls.unescape_string(m[1])
                value = cls.unescape_string(m[2])
                if strings.get(key, value) != value:
                    cls.logger.error(
                        "multiple different definitions of %r in %s: %r, %r; overwriting",
                        key,
                        path,
                        strings[key],
                        value,
                    )
                elif key in strings:
                    assert strings[key] == value
                    cls.logger.warning("duplicate string definition %r in %s", key, path)
                strings[key] = value
        return strings


class Data(TypedDict):
    tires: Dict[str, List[TruckTire]]
    trucks: Dict[str, List[Truck]]
    strings: Dict[str, str]


def load_data(initial_pak: str) -> Data:
    pak_path = pathlib.Path(initial_pak)
    with zipfile.ZipFile(pak_path, "r") as pak:
        path = pathlib.PureWindowsPath("[media]") / "_templates" / "trucks.xml"
        with pak.open(str(path), "r") as fp:
            with load_xml_with_error(fp.read(), pak_path, path) as roots:
                truck_templates = load_templates(roots)

        tires = load_all_tires(pak, truck_templates)
        engines = load_all_engines(pak, {})
        return {
            "tires": tires,
            "trucks": load_all_trucks(
                pak,
                {},
                engines=engines,
                tires=tires,
            ),
            "strings": Strings.load(pak, pathlib.PureWindowsPath("[strings]") / "strings_english.str"),
        }


"""
class XMLStructure(TypedDict):
    attribs: Set[str]
    children: Set[str]
    parents: Set[str]
    src: Dict[str, Set[str]]


def get_xml_structure(root: str) -> Dict[str, XMLStructure]:
    structure = {}  # type: XMLStructure
    for elems, pak, name in iter_xml_data(root):
        for e in elems:
            for e in e.iter():
                for k, v in e.items():
                    x = structure.setdefault(e.tag, {})
                    x.setdefault("attribs", set()).add(k)
                    for c in e.iterfind("./*"):
                        x.setdefault("children", set()).add(c.tag)
                        structure.setdefault(c.tag, {}).setdefault(
                            "parents", set()
                        ).add(e.tag)
                    x.setdefault("src", {}).setdefault(pak.name, set()).add(str(name))
    return structure
"""


def xml_escape(x: str) -> str:
    def repl(m: re.Match) -> str:
        return "&#%d;" % ord(m[0])

    return re.sub(r"[^\n !#$%(-;=?-~]", repl, x)


def json_decimal_default(x):
    if isinstance(x, Decimal):
        return float(x)
    else:
        raise TypeError


def draw_engine_chart(engines: List[Engine], strings: Dict[str, str]) -> str:
    labels = []
    torques = []
    for e in sorted(engines, key=lambda e: e.Torque):
        labels.append(strings.get(e.UiName, e.UiName))
        torques.append(e.Torque)
    if not labels:
        return ""
    id = str(uuid.uuid4())
    return '''<h3>Engines</h3>
<div class="engine-chart"><canvas id="%(id)s"></canvas><script>
"use strict";
document.addEventListener("DOMContentLoaded", async () => {
    const elem = document.getElementById(%(json_id)s);
    new Chart(elem, %(json_data)s);
});
</script></div>''' % {
    "id": id,
    "json_id": json.dumps(id),
    "json_data": json.dumps(
        {
            "type": "bar",
            "data": {
                "labels": labels,
                "datasets": [
                    {
                        "label": "Torque",
                        "data": torques,
                    },
                ],
            },
            "options": {
                "scales": {
                    "y": {
                        "beginAtZero": True,
                    },
                },
            },
        },
        default=json_decimal_default,
    ),
}


def draw_tire_chart(tires: Iterable[TruckTire], strings: Dict[str, str]) -> str:
    maximum = Decimal(3)
    grouped_tires = {}  # type: Dict[Tuple[Decimal, Decimal, Decimal], List[str]]
    for wheel in truck.CompatibleWheels:
        for t in wheel.Tires:
            k = (
                t.WheelFriction.SubstanceFriction,
                t.WheelFriction.BodyFriction,
                t.WheelFriction.BodyFrictionAsphalt,
            )
            grouped_tires.setdefault(k, set()).add(
                strings.get(t.UiName, t.UiName),
            )
            maximum = max(maximum, *k)
    if not grouped_tires:
        return ""

    series = [
        { "label": ", ".join(sorted(names)), "data": data, }
        for data, names in reversed(sorted(grouped_tires.items()))
    ]

    xml = ['<h3>Tires</h3>\n<div class="tire-charts">']
    for dataset in series:
        id = str(uuid.uuid4())
        xml.append(
            '<div><canvas id="%(id)s"></canvas><script>drawChart(%(args)s);</script></div>' % {
                "id": id,
                "args": json.dumps(
                    {
                        "id": id,
                        "datasets": [dataset],
                        "max": maximum,
                    },
                    default=json_decimal_default,
                ),
            }
        )
    xml.append("</div>")
    return "".join(xml)


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        stream=sys.stderr,
        level=logging.WARNING,
        format="%(name)s %(levelname)-8s :%(lineno)-3d %(message)s",
    )
    # there are lots of "multiple different definitions"
    Strings.logger.setLevel(logging.ERROR + 1)

    data = load_data(
        "/data/sata/steam/SteamLibrary/steamapps/common/SnowRunner/preload/paks/client/initial.pak"
    )
    with open(sys.stdout.fileno(), "w", encoding="ascii", closefd=False) as fp:
        fp.write(
            '''<style>
    .engine-chart, .tire-charts {
        min-height: 8em;
        max-height: 25vh;
    }
    .tire-charts {
        display: flex;
        overflow-x: scroll;
    }
    details {
        border-left: .5em solid lightgray;
        padding-left: .5em;
    }
</style>
'''
        )
        fp.write('<script src="https://cdn.jsdelivr.net/npm/chart.js" defer=""></script>')
        fp.write("<script>\n")
        fp.write(
'''"use strict";
function drawChart({ id, datasets, max }) {
    document.addEventListener("DOMContentLoaded", async () => {
        const elem = document.getElementById(id);
        new Chart(elem, {
            type: "radar",
            data: {
                labels: ["Mud", "Dirt", "Asphalt"],
                datasets: datasets,
            },
            options: {
                scales: {
                    r: {
                        min: 0,
                        max: Math.ceil(max),
                    },
                },
            },
        });
    });
}'''
        )
        fp.write("\n</script>")
        fp.write('<h1 id="trucks">')
        fp.write(xml_escape("Trucks"))
        fp.write("</h1>\n")

        for display_name, _, truck in sorted(
            (data["strings"][t.UiName], i, t)
            for i, t in enumerate(
                itertools.chain.from_iterable(data["trucks"].values())
            )
        ):
            fp.write("<h2>")
            fp.write(xml_escape(display_name))
            fp.write("</h2>\n")
            fp.write('<details open="">\n')
            fp.write("<details>\n")
            fp.write("<pre>")
            fp.write(xml_escape(pprint.pformat(truck)))
            fp.write("</pre>\n")
            fp.write("</details>\n")
            fp.write(
                draw_tire_chart(
                    itertools.chain.from_iterable(
                        w.Tires for w in truck.CompatibleWheels
                    ),
                    data["strings"],
                )
            )
            fp.write(
                draw_engine_chart(
                    itertools.chain.from_iterable(
                        es.Engines for es in truck.EngineSocket
                    ),
                    data["strings"],
                )
            )
            fp.write("</details>\n")
