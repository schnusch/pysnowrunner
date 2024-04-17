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
import itertools
import logging
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal
from typing import (
    Any,
    ClassVar,
    Dict,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Type,
    TypeVar,
)

logger = logging.getLogger(__name__)
T_Base = TypeVar("T_Base", bound="_Base")

TemplateDict = Dict[Type[T_Base], Dict[str, T_Base]]


class _Base(object):
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
        templates: TemplateDict,
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


@dataclass
class Base(_Base):
    def asdict_non_recursive(self) -> Dict[str, Any]:
        return dict((f.name, getattr(self, f.name)) for f in dataclasses.fields(self))


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
        templates: TemplateDict,
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
        templates: TemplateDict,
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
        templates: TemplateDict,
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
        templates: TemplateDict,
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
        templates: TemplateDict,
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
        templates: TemplateDict,
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
        templates: TemplateDict,
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


T_WheelTemplate = TypeVar("T_WheelTemplate", bound="WheelTemplate")


@dataclass
class WheelTemplate(DecimalAttributes):
    tag = "Wheel"
    decimal_attributes = ("SteeringAngle", "SteeringCastor")

    Location: Optional[str]
    Torque: str
    ConnectedToHandbrake: bool = dataclasses.field(default=False, kw_only=True)
    SteeringAngle: Decimal = dataclasses.field(default=Decimal(0), kw_only=True)
    SteeringCastor: Decimal = dataclasses.field(default=Decimal(0), kw_only=True)

    @classmethod
    def from_xml(
        cls: Type[T_WheelTemplate],
        elem: ET.Element,
        templates: TemplateDict,
    ) -> T_WheelTemplate:
        self = super().from_xml(elem, templates)

        handbrake = elem.get("ConnectedToHandbrake")
        if handbrake is not None and handbrake != "false":
            assert handbrake == "true"
            self = dataclasses.replace(self, ConnectedToHandbrake=True)

        location = elem.get("Location")
        if location is not None:
            self = dataclasses.replace(self, Location=location)

        torque = elem.get("Torque")
        if torque is not None:
            self = dataclasses.replace(self, Torque=torque)

        return self


class PosTuple(NamedTuple):
    x: Decimal
    y: Decimal
    z: Decimal


T_Wheel = TypeVar("T_Wheel", bound="Wheel")


@dataclass
class Wheel(WheelTemplate):
    Pos: PosTuple
    RightSide: bool = dataclasses.field(default=False, kw_only=True)

    @classmethod
    def from_xml(
        cls: Type[T_Wheel],
        elem: ET.Element,
        templates: TemplateDict,
    ) -> T_Wheel:
        self = super().from_xml(elem, templates)

        pos = elem.get("Pos")
        if pos is not None:
            assert pos.startswith("(") and pos.endswith(")")
            parsed = tuple(Decimal(x) for x in pos[1:-1].split(";"))
            assert len(parsed) == 3
            self = dataclasses.replace(self, Pos=PosTuple(*parsed))

        right_side = elem.get("RightSide")
        if right_side is not None and right_side != "false":
            assert right_side == "true"
            self = dataclasses.replace(self, RightSide=True)

        assert (
            self.Pos is not None and self.RightSide is not None
        ), "%r has None attributes: %s" % (self, ET.tostring(elem, encoding="unicode"))
        return self


@dataclass
class TWheelBase:
    left: Decimal
    all: Decimal
    right: Decimal


T_Truck = TypeVar("T_Truck", bound="Truck")


@dataclass
class Truck(Base):
    tag = "Truck"

    UiName: str
    Price: int
    FuelCapacity: int
    CompatibleWheels: List[CompatibleWheels]
    EngineSocket: List[EngineSocket]
    Wheels: List[Wheel]
    ExtraWheels: List[Wheel]
    WheelBase: TWheelBase
    TrackWidths: List[Tuple[Decimal, Wheel, Wheel]]

    @staticmethod
    def iter_wheelset_pairs(wheels: List[Wheel]) -> Iterator[Tuple[Wheel, Wheel]]:
        """Find the closest non-right wheel to every nonright wheel or
        vice versa and yield a tuple of them."""
        left_wheels = [w for w in wheels if not w.RightSide]
        right_wheels = [w for w in wheels if w.RightSide]
        if len(left_wheels) < len(right_wheels):
            main = left_wheels
            other = right_wheels
        else:
            main = right_wheels
            other = left_wheels
        """
        logger.error(
            "iter_wheelset_pairs:\n%s\n%s",
            pprint.pformat(left_wheels),
            pprint.pformat([w for w in wheels if w.RightSide]),
        )
        """
        for main_wheel in main:
            closest_other_wheel = -1
            for i, other_wheel in enumerate(other):
                if closest_other_wheel < 0 or (
                    abs(other_wheel.Pos.x - main_wheel.Pos.x)
                    < abs(other[closest_other_wheel].Pos.x - main_wheel.Pos.x)
                ):
                    closest_other_wheel = i
            assert closest_other_wheel >= 0
            left_wheel = main_wheel
            right_wheel = other.pop(closest_other_wheel)
            if left_wheel.RightSide:
                left_wheel, right_wheel = right_wheel, left_wheel
            assert not left_wheel.RightSide and right_wheel.RightSide
            yield (left_wheel, right_wheel)

    def iter_wheel_pairs(self) -> Iterator[Tuple[Wheel, Wheel]]:
        for wheels in (self.Wheels, self.ExtraWheels):
            yield from self.iter_wheelset_pairs(wheels)

    def get_wheel_base(self) -> TWheelBase:
        ax = [w.Pos.x for w in itertools.chain(self.Wheels, self.ExtraWheels)]
        lx = []
        rx = []
        for l, r in self.iter_wheel_pairs():  # noqa: E741
            lx.append(l.Pos.x)
            rx.append(r.Pos.x)
        return TWheelBase(
            left=max(lx) - min(lx) if lx else Decimal(0),
            all=max(ax) - min(ax) if ax else Decimal(0),
            right=max(rx) - min(rx) if rx else Decimal(0),
        )

    def get_track_widths(self) -> List[Tuple[Decimal, Wheel, Wheel]]:
        return [
            # both y positions seem to be positive and are negated by RightSide attribute
            (lw.Pos.y + rw.Pos.y, lw, rw)
            for lw, rw in sorted(
                self.iter_wheel_pairs(),
                key=lambda ws: -ws[0].Pos.x,
            )
        ]

    @classmethod
    def from_xml(  # type: ignore[override]
        cls: Type[T_Truck],
        elem: ET.Element,
        templates: TemplateDict,
        engines: Dict[str, Dict[str, Engine]],
        tires: Dict[str, Dict[str, TruckTire]],
    ) -> T_Truck:
        self = super().from_xml(elem, templates)

        truck_data = elem.find("./TruckData")
        if truck_data is not None:
            fuel_capacity = truck_data.get("FuelCapacity")
            if fuel_capacity is not None:
                self = dataclasses.replace(
                    self,
                    FuelCapacity=int(fuel_capacity, 10),
                )

            self = dataclasses.replace(
                self,
                CompatibleWheels=[
                    CompatibleWheels.from_xml(w, templates, tires)
                    for w in truck_data.iterfind("./CompatibleWheels")
                ],
                EngineSocket=[
                    EngineSocket.from_xml(e, templates, engines)
                    for e in truck_data.iterfind("./EngineSocket")
                ],
                Wheels=[
                    Wheel.from_xml(w, templates)
                    for w in truck_data.iterfind("./Wheels/Wheel")
                ],
                ExtraWheels=[
                    Wheel.from_xml(w, templates)
                    for w in truck_data.iterfind("./ExtraWheels/Wheel")
                ],
            )

            self = dataclasses.replace(self, WheelBase=self.get_wheel_base())

            self = dataclasses.replace(self, TrackWidths=self.get_track_widths())
        else:
            self = dataclasses.replace(
                self,
                CompatibleWheels=[],
                EngineSocket=[],
                Wheels=[],
                ExtraWheels=[],
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

        assert self.UiName is not None, "%r has None attributes: %s" % (
            self,
            ET.tostring(elem, encoding="unicode"),
        )
        return self
