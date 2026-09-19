"""The fictional merchant's product domain.

One Indian apparel and fashion house, described in enough
depth that compatibility, cross-sell and merchandising questions have real
answers. Depth matters more than row count (ADR 0025): every line below exists to
make some question answerable, not to inflate a total.

Nothing here is random. The generator walks these tables deterministically, so the
same version and seed always produce the same catalog.
"""

from __future__ import annotations

from dataclasses import dataclass, field

CURRENCY = "INR"

# Capabilities a variant can offer, and that another variant can require. These
# are the only facts `check_compatibility` is allowed to reason from (ADR 0006).
CAPABILITIES: tuple[tuple[str, str, str], ...] = (
    ("cap_fabric", "Primary fabric or material", "text"),
    ("cap_size_system", "Sizing system the piece is cut to", "text"),
    ("cap_fit", "Cut or fit", "text"),
    ("cap_care", "Care the material needs", "text"),
    ("cap_outfit_set", "Outfit set the piece is tailored to match", "text"),
    ("cap_mount", "Fitting standard (strap lug, buckle, insole)", "text"),
)


@dataclass(frozen=True)
class Line:
    """One product line: a family of closely related SKUs."""

    key: str
    name: str
    category: str
    # (low, high) list price in paise, before the generator picks a point in range.
    price_range: tuple[int, int]
    variant_axis: str                     # what distinguishes the variants
    variant_values: tuple[str, ...]
    specs: tuple[tuple[str, object, str | None], ...] = ()   # (key, value, unit)
    provides: tuple[tuple[str, object], ...] = ()            # capability -> value
    requires: tuple[tuple[str, str, object, str], ...] = ()  # (cap, op, value, why)
    pairs_with: tuple[str, ...] = ()      # other line keys this genuinely complements
    blurb: str = ""


CATEGORIES: tuple[tuple[str, str, str | None], ...] = (
    ("cat_menswear", "Menswear", None),
    ("cat_womenswear", "Womenswear", None),
    ("cat_ethnic", "Ethnic Wear", None),
    ("cat_ethnic_coord", "Ethnic Coordinates", "cat_ethnic"),
    ("cat_denim", "Denim", None),
    ("cat_activewear", "Activewear", None),
    ("cat_footwear", "Footwear", None),
    ("cat_accessories", "Accessories", None),
    ("cat_care", "Garment & Shoe Care", None),
)

# The house labels. A single fashion house with its own labels, not a marketplace.
BRANDS: tuple[str, ...] = ("Aster", "Meridian", "Solace", "Nimbus", "Aldervale", "Kestrel")

# Signature outfit sets the coordinate pieces (blouses, dupattas) are tailored to.
# Those pieces reference a set by capability, which is what makes "will this blouse
# match my saree?" answerable.
OUTFIT_SETS: tuple[str, ...] = (
    "Kanjeevaram Heritage", "Banarasi Royale", "Chanderi Dawn", "Bandhani Festive",
    "Ikat Monsoon",
)

# The category whose pieces are cut for exactly one outfit set.
MATCHED_CATEGORY = "cat_ethnic_coord"

# Fitting standards shared between a base piece and its add-on. Both sides carry the
# value, so an add-on can be checked against the piece it attaches to.
MOUNTS: dict[str, str] = {
    "watch_analog": "lug_20mm",
    "watch_strap": "lug_20mm",
    "sneakers": "insole_standard",
    "running_shoes": "insole_standard",
    "insoles": "insole_standard",
    "belt_leather": "buckle_35mm",
    "belt_buckle": "buckle_35mm",
}

LINES: tuple[Line, ...] = (
    # ------------------------------------------------------------------ menswear
    Line(
        key="oxford_shirt", name="Oxford Shirt", category="cat_menswear",
        price_range=(129900, 349900), variant_axis="colour",
        variant_values=("White", "Sky Blue"),
        specs=(("gsm", 140, "gsm"), ("cotton_pct", 100, "%"), ("wrinkle_free", True, None)),
        provides=(("cap_fabric", "cotton"), ("cap_size_system", "in_apparel"), ("cap_fit", "slim")),
        pairs_with=("chinos", "belt_leather"),
        blurb="A breathable oxford-weave shirt that holds a collar through a humid day.",
    ),
    Line(
        key="linen_shirt", name="Linen Shirt", category="cat_menswear",
        price_range=(179900, 449900), variant_axis="colour",
        variant_values=("Natural", "Sage"),
        specs=(("gsm", 160, "gsm"), ("linen_pct", 100, "%")),
        provides=(("cap_fabric", "linen"), ("cap_care", "gentle_wash")),
        pairs_with=("chinos", "loafers"),
        blurb="Pure linen for Indian summers, cut relaxed through the body.",
    ),
    Line(
        key="polo_tshirt", name="Polo T-Shirt", category="cat_menswear",
        price_range=(79900, 199900), variant_axis="colour",
        variant_values=("Black", "Heather Grey"),
        specs=(("gsm", 220, "gsm"), ("pique_knit", True, None)),
        provides=(("cap_fabric", "cotton"),),
        pairs_with=("chinos", "sneakers"),
        blurb="A heavyweight piqué polo that keeps its shape after fifty washes.",
    ),
    Line(
        key="chinos", name="Stretch Chinos", category="cat_menswear",
        price_range=(149900, 399900), variant_axis="fit",
        variant_values=("Slim", "Tapered", "Relaxed"),
        specs=(("stretch_pct", 2, "%"), ("gsm", 260, "gsm")),
        provides=(("cap_fabric", "cotton"),),
        pairs_with=("belt_leather", "loafers"),
        blurb="Office-to-weekend chinos with just enough stretch to sit cross-legged.",
    ),
    Line(
        key="blazer", name="Unstructured Blazer", category="cat_menswear",
        price_range=(599900, 1499900), variant_axis="colour",
        variant_values=("Charcoal", "Navy"),
        specs=(("lined", False, None), ("wool_pct", 60, "%")),
        provides=(("cap_fabric", "wool_blend"), ("cap_care", "dry_clean")),
        pairs_with=("oxford_shirt", "chinos"),
        blurb="A soft-shouldered blazer light enough for a Mumbai evening.",
    ),
    # ---------------------------------------------------------------- womenswear
    Line(
        key="wrap_dress", name="Wrap Dress", category="cat_womenswear",
        price_range=(199900, 599900), variant_axis="print",
        variant_values=("Floral", "Solid"),
        specs=(("length", "midi", None), ("viscose_pct", 100, "%")),
        provides=(("cap_fabric", "viscose"), ("cap_care", "gentle_wash")),
        pairs_with=("block_heels", "tote_bag"),
        blurb="A flowing midi wrap dress that adjusts to fit rather than cling.",
    ),
    Line(
        key="tailored_trousers", name="Tailored Trousers", category="cat_womenswear",
        price_range=(179900, 449900), variant_axis="colour",
        variant_values=("Black", "Camel"),
        specs=(("high_rise", True, None), ("pockets", 2, None)),
        provides=(("cap_fabric", "poly_viscose"),),
        pairs_with=("silk_blouse", "block_heels"),
        blurb="High-rise pleated trousers with real pockets.",
    ),
    Line(
        key="silk_blouse", name="Silk Blouse", category="cat_womenswear",
        price_range=(249900, 699900), variant_axis="colour",
        variant_values=("Champagne", "Emerald"),
        specs=(("silk_pct", 100, "%"), ("momme", 19, "mm")),
        provides=(("cap_fabric", "silk"), ("cap_care", "dry_clean")),
        pairs_with=("tailored_trousers",),
        blurb="Washed mulberry silk that drapes rather than shines.",
    ),
    Line(
        key="knit_cardigan", name="Knit Cardigan", category="cat_womenswear",
        price_range=(199900, 499900), variant_axis="colour",
        variant_values=("Oatmeal", "Dusty Rose"),
        specs=(("gauge", 12, None), ("acrylic_free", True, None)),
        provides=(("cap_fabric", "cotton_blend"), ("cap_care", "gentle_wash")),
        pairs_with=("wrap_dress",),
        blurb="A light cardigan for over-air-conditioned offices.",
    ),
    # --------------------------------------------------------------- ethnic wear
    Line(
        key="kurta_set", name="Kurta Set", category="cat_ethnic",
        price_range=(249900, 899900), variant_axis="colour",
        variant_values=("Indigo", "Mustard", "Rani Pink"),
        specs=(("pieces", 3, None), ("hand_block_print", True, None)),
        provides=(("cap_fabric", "cotton"),),
        pairs_with=("juttis", "jhumkas"),
        blurb="A hand block-printed kurta, palazzo and dupatta set.",
    ),
    Line(
        key="saree", name="Silk Saree", category="cat_ethnic",
        price_range=(699900, 2999900), variant_axis="weave",
        variant_values=("Kanjeevaram", "Banarasi", "Chanderi"),
        specs=(("length_m", 6.3, "m"), ("zari", True, None), ("blouse_piece", True, None)),
        provides=(("cap_fabric", "silk"), ("cap_care", "dry_clean")),
        pairs_with=("saree_blouse", "jhumkas"),
        blurb="Handloom silk with real zari, woven in the region it is named for.",
    ),
    Line(
        key="lehenga", name="Lehenga Choli", category="cat_ethnic",
        price_range=(1499900, 4999900), variant_axis="colour",
        variant_values=("Crimson", "Teal", "Blush"),
        specs=(("flare_m", 4, "m"), ("can_can", True, None), ("embroidered", True, None)),
        provides=(("cap_fabric", "georgette"), ("cap_care", "dry_clean")),
        pairs_with=("dupatta", "jhumkas"),
        blurb="A festive lehenga with hand embroidery that moves when you do.",
    ),
    Line(
        key="nehru_jacket", name="Nehru Jacket", category="cat_ethnic",
        price_range=(199900, 599900), variant_axis="colour",
        variant_values=("Ivory", "Midnight"),
        specs=(("jacquard", True, None),),
        provides=(("cap_fabric", "silk_blend"), ("cap_care", "dry_clean")),
        pairs_with=("kurta_set", "juttis"),
        blurb="A textured bandhgala jacket that lifts a plain kurta for a wedding.",
    ),
    # --------------------------------------------------------- ethnic coordinates
    Line(
        key="saree_blouse", name="Designer Blouse", category="cat_ethnic_coord",
        price_range=(149900, 499900), variant_axis="sleeve",
        variant_values=("Sleeveless", "Elbow"),
        specs=(("padded", True, None), ("back_hook", True, None)),
        requires=(("cap_outfit_set", "eq", None,
                   "A designer blouse is tailored to one outfit set and will not match another."),),
        blurb="A ready-to-wear blouse colour-matched to one specific saree set.",
    ),
    Line(
        key="dupatta", name="Embroidered Dupatta", category="cat_ethnic_coord",
        price_range=(99900, 349900), variant_axis="fabric",
        variant_values=("Organza", "Chiffon"),
        specs=(("length_m", 2.5, "m"), ("tasselled", True, None)),
        requires=(("cap_outfit_set", "eq", None,
                   "A dupatta is dyed to one outfit set."),),
        blurb="A dupatta dyed and embroidered to finish one particular set.",
    ),
    # --------------------------------------------------------------------- denim
    Line(
        key="jeans_slim", name="Slim Jeans", category="cat_denim",
        price_range=(179900, 499900), variant_axis="wash",
        variant_values=("Raw", "Mid Wash", "Black"),
        specs=(("denim_oz", 12, "oz"), ("stretch_pct", 2, "%"), ("selvedge", False, None)),
        provides=(("cap_fabric", "denim"),),
        pairs_with=("denim_jacket", "sneakers", "belt_leather"),
        blurb="Stretch selvedge-look denim that fades in the right places.",
    ),
    Line(
        key="denim_jacket", name="Denim Jacket", category="cat_denim",
        price_range=(249900, 699900), variant_axis="wash",
        variant_values=("Light", "Mid"),
        specs=(("denim_oz", 13, "oz"), ("sherpa_lined", False, None)),
        provides=(("cap_fabric", "denim"),),
        pairs_with=("jeans_slim", "polo_tshirt"),
        blurb="A trucker jacket that softens with every wear.",
    ),
    # --------------------------------------------------------------- activewear
    Line(
        key="leggings", name="Training Leggings", category="cat_activewear",
        price_range=(99900, 299900), variant_axis="colour",
        variant_values=("Black", "Plum"),
        specs=(("squat_proof", True, None), ("pockets", 2, None)),
        provides=(("cap_fabric", "nylon_spandex"),),
        pairs_with=("sports_bra", "running_shoes"),
        blurb="Squat-proof leggings with a phone pocket that actually holds a phone.",
    ),
    Line(
        key="sports_bra", name="Sports Bra", category="cat_activewear",
        price_range=(79900, 249900), variant_axis="support",
        variant_values=("Low", "Medium", "High"),
        specs=(("removable_pads", True, None), ("moisture_wicking", True, None)),
        provides=(("cap_fabric", "nylon_spandex"),),
        pairs_with=("leggings",),
        blurb="Support rated honestly for the workout it is sold for.",
    ),
    Line(
        key="dri_fit_tee", name="Performance Tee", category="cat_activewear",
        price_range=(59900, 179900), variant_axis="colour",
        variant_values=("Graphite", "White"),
        specs=(("moisture_wicking", True, None), ("upf", 30, None)),
        provides=(("cap_fabric", "polyester"),),
        pairs_with=("running_shoes",),
        blurb="A quick-dry running tee for a Chennai morning.",
    ),
    # ----------------------------------------------------------------- footwear
    Line(
        key="sneakers", name="Leather Sneakers", category="cat_footwear",
        price_range=(299900, 899900), variant_axis="colour",
        variant_values=("White", "Off-White", "Black"),
        specs=(("sole", "rubber cupsole", None), ("weight_g", 420, "g")),
        provides=(("cap_fabric", "leather"), ("cap_size_system", "uk_shoe")),
        pairs_with=("shoe_care_kit", "insoles"),
        blurb="Minimal full-grain leather sneakers on a stitched cupsole.",
    ),
    Line(
        key="running_shoes", name="Running Shoes", category="cat_footwear",
        price_range=(349900, 1199900), variant_axis="colour",
        variant_values=("Blue", "Grey"),
        specs=(("drop_mm", 8, "mm"), ("weight_g", 260, "g"), ("carbon_plate", False, None)),
        provides=(("cap_fabric", "mesh"), ("cap_size_system", "uk_shoe")),
        pairs_with=("insoles", "dri_fit_tee"),
        blurb="Cushioned daily trainers built for Indian road running.",
    ),
    Line(
        key="loafers", name="Suede Loafers", category="cat_footwear",
        price_range=(249900, 699900), variant_axis="colour",
        variant_values=("Tan", "Chocolate"),
        specs=(("hand_stitched", True, None),),
        provides=(("cap_fabric", "suede"), ("cap_size_system", "uk_shoe")),
        pairs_with=("shoe_care_kit", "chinos"),
        blurb="Unlined suede loafers that break in within a week.",
    ),
    Line(
        key="juttis", name="Embroidered Juttis", category="cat_footwear",
        price_range=(99900, 299900), variant_axis="colour",
        variant_values=("Gold", "Maroon"),
        specs=(("cushioned", True, None), ("handmade", True, None)),
        provides=(("cap_fabric", "leather"), ("cap_size_system", "uk_shoe")),
        pairs_with=("kurta_set",),
        blurb="Hand-embroidered Punjabi juttis with a cushioned footbed.",
    ),
    Line(
        key="block_heels", name="Block Heels", category="cat_footwear",
        price_range=(179900, 499900), variant_axis="height",
        variant_values=("2 in", "3 in"),
        specs=(("padded_insole", True, None),),
        provides=(("cap_size_system", "uk_shoe"),),
        pairs_with=("wrap_dress",),
        blurb="A block heel you can stand in through a whole sangeet.",
    ),
    Line(
        key="insoles", name="Comfort Insoles", category="cat_footwear",
        price_range=(49900, 149900), variant_axis="arch",
        variant_values=("Neutral", "High Arch"),
        specs=(("memory_foam", True, None),),
        requires=(("cap_mount", "eq", "insole_standard",
                   "These insoles fit a removable standard footbed only."),),
        blurb="Memory-foam insoles for shoes with a removable footbed.",
    ),
    # -------------------------------------------------------------- accessories
    Line(
        key="belt_leather", name="Leather Belt", category="cat_accessories",
        price_range=(79900, 249900), variant_axis="colour",
        variant_values=("Black", "Brown"),
        specs=(("width_mm", 35, "mm"), ("full_grain", True, None)),
        provides=(("cap_fabric", "leather"),),
        pairs_with=("belt_buckle", "shoe_care_kit"),
        blurb="A full-grain belt with an interchangeable 35 mm buckle.",
    ),
    Line(
        key="belt_buckle", name="Belt Buckle", category="cat_accessories",
        price_range=(49900, 149900), variant_axis="finish",
        variant_values=("Brushed Steel", "Antique Brass"),
        specs=(("solid_brass", True, None),),
        requires=(("cap_mount", "eq", "buckle_35mm",
                   "This buckle fits a 35 mm interchangeable strap only."),),
        blurb="A swap-in buckle for the house belts.",
    ),
    Line(
        key="watch_analog", name="Analog Watch", category="cat_accessories",
        price_range=(299900, 1499900), variant_axis="dial",
        variant_values=("Black", "Silver", "Green"),
        specs=(("case_mm", 40, "mm"), ("sapphire", True, None), ("water_resistance", "5ATM", None)),
        pairs_with=("watch_strap",),
        blurb="A minimal automatic-look dial on a 20 mm quick-release strap.",
    ),
    Line(
        key="watch_strap", name="Watch Strap", category="cat_accessories",
        price_range=(49900, 199900), variant_axis="material",
        variant_values=("Leather", "Canvas", "Mesh"),
        specs=(("quick_release", True, None),),
        requires=(("cap_mount", "eq", "lug_20mm",
                   "This strap fits a 20 mm quick-release lug only."),),
        blurb="A quick-release strap in the width the house watches use.",
    ),
    Line(
        key="tote_bag", name="Leather Tote", category="cat_accessories",
        price_range=(299900, 999900), variant_axis="colour",
        variant_values=("Tan", "Black"),
        specs=(("laptop_sleeve", True, None), ("capacity_l", 14, "L")),
        provides=(("cap_fabric", "leather"),),
        pairs_with=("shoe_care_kit",),
        blurb="A structured tote that fits a 14-inch laptop and lunch.",
    ),
    Line(
        key="sunglasses", name="Sunglasses", category="cat_accessories",
        price_range=(149900, 599900), variant_axis="frame",
        variant_values=("Aviator", "Wayfarer"),
        specs=(("polarised", True, None), ("uv400", True, None)),
        blurb="Polarised UV400 lenses in acetate or metal frames.",
    ),
    Line(
        key="jhumkas", name="Jhumka Earrings", category="cat_accessories",
        price_range=(59900, 249900), variant_axis="finish",
        variant_values=("Oxidised Silver", "Gold Plated"),
        specs=(("hypoallergenic", True, None),),
        pairs_with=("kurta_set", "saree"),
        blurb="Temple-style jhumkas light enough to wear all evening.",
    ),
    # ----------------------------------------------------------------- care
    Line(
        key="shoe_care_kit", name="Leather Care Kit", category="cat_care",
        price_range=(49900, 149900), variant_axis="kit",
        variant_values=("Clean & Condition", "Full Restore"),
        specs=(("brush_included", True, None),),
        requires=(("cap_fabric", "eq", "leather",
                   "This kit is for smooth leather; it will stain suede, canvas or mesh."),),
        blurb="Cleaner, conditioner and horsehair brush for smooth leather.",
    ),
)

# Demo stories: a handful of variants given a deliberate stock and sales shape, so
# every merchant edge case has a real subject instead of depending on the random
# walk. Each names the first edition of a line and one of its variant values. Sales
# land as ordinary paid journeys inside the last thirty days before `as_of`, which is
# the window the merchant's cover and pricing reads use.
#   (line key, variant index, units sold in the last 30 days,
#    on-hand per location in LOCATIONS order, what the story demonstrates)
DEMAND_STORIES: tuple[tuple[str, int, int, tuple[int, int, int], str], ...] = (
    ("saree", 0, 60, (3, 3, 2), "Festive best seller with about four days of cover"),
    ("sneakers", 0, 36, (4, 3, 3), "Steady seller just under the ten-day alert line"),
    ("kurta_set", 0, 30, (0, 0, 0), "Best seller already out of stock everywhere"),
    ("running_shoes", 0, 24, (0, 0, 7), "Selling well, but all remaining stock sits in Mumbai"),
    ("oxford_shirt", 0, 45, (70, 70, 60), "Top seller with healthy stock: no alert"),
)

# Dead stock: deep inventory and no sales at all, the markdown / pricing-headroom
# subject. These variants are kept out of the generated journeys entirely.
#   (line key, variant index, on-hand per location)
DEAD_STOCK: tuple[tuple[str, int, tuple[int, int, int]], ...] = (
    ("blazer", 0, (50, 50, 50)),
    ("block_heels", 1, (40, 30, 30)),
)

# A live markdown: a promotional price below list, with the list price shown as the
# compare-at, so a pricing question has a discount history to read.
#   (line key, variant index, promotional price as a fraction of list)
MARKDOWNS: tuple[tuple[str, int, float], ...] = (
    ("denim_jacket", 0, 0.8),
)

# Product-line adjectives, walked deterministically to give each SKU a distinct
# name without a random word soup.
EDITIONS: tuple[str, ...] = (
    "Core", "Studio", "Pro", "Air", "Max", "Lite", "Everyday", "Signature",
    "Field", "Metro", "Halo", "Atlas", "Aurora", "Quartz", "Slate", "Lumen",
)

LOCATIONS: tuple[tuple[str, str, str, str], ...] = (
    ("loc_blr", "BLR", "Bengaluru fulfilment centre", "South"),
    ("loc_del", "DEL", "Delhi NCR fulfilment centre", "North"),
    ("loc_mum", "MUM", "Mumbai fulfilment centre", "West"),
)

# Given names and surnames for seeded customers. Indian-market appropriate, and
# fixed so a reset reproduces the same people.
GIVEN_NAMES: tuple[str, ...] = (
    "Ira", "Dev", "Anaya", "Kabir", "Meera", "Rohan", "Sana", "Vikram", "Priya", "Arjun",
    "Nikhil", "Tara", "Aditya", "Kavya", "Rahul", "Divya", "Farhan", "Neha", "Siddharth", "Riya",
)
SURNAMES: tuple[str, ...] = (
    "Menon", "Rao", "Sharma", "Iyer", "Banerjee", "Chawla", "Nair", "Kulkarni", "Desai", "Bose",
)

FULFILMENT_OPTIONS: tuple[tuple[str, int, int], ...] = (
    # (option, shipping in paise, promised days)
    ("standard", 0, 4),
    ("express", 9900, 2),
)

PROMOTIONS: tuple[tuple[str, str, str, int, int, str | None], ...] = (
    # (code, description, kind, value, min subtotal in paise, category scope)
    # A promotion has to be enforceable exactly as its description reads: the
    # category is what confines the ethnic-wear and footwear offers to the aisles they
    # name, and a null scope is genuinely storewide.
    ("MONSOON10", "Monsoon sale: ₹200 off orders above ₹2,000", "fixed_minor", 20000,
     200000, None),
    ("FESTIVE500", "₹500 off ethnic wear above ₹5,000", "fixed_minor", 50000,
     500000, "cat_ethnic"),
    ("STRIDE15", "15% off footwear above ₹3,000", "percentage", 15,
     300000, "cat_footwear"),
)

CAMPAIGNS: tuple[tuple[str, str, str, int], ...] = (
    # (name, channel, promotion code, budget in paise)
    ("Festive Ethnic Push", "search", "FESTIVE500", 15000000),
    ("Stride Into Diwali", "social", "STRIDE15", 25000000),
    ("Always-On Brand", "display", "MONSOON10", 8000000),
)


@dataclass
class Counts:
    """What one generator run produced, for the reset acceptance check."""

    values: dict[str, int] = field(default_factory=dict)

    def add(self, key: str, n: int = 1) -> None:
        self.values[key] = self.values.get(key, 0) + n

    def as_dict(self) -> dict[str, int]:
        return dict(sorted(self.values.items()))
