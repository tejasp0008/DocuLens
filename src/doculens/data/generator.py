"""Synthetic Indian-style insurance claim documents with exact ground truth.

Real claim documents cannot be published with a course report, so DocuLens
generates its own corpus: motor-insurance claim forms and garage repair
invoices populated with Faker's ``en_IN`` provider. Because the text is
rendered rather than scanned, the ground truth is known exactly, which is what
makes CER and WER meaningful.

Each document is written three ways: a PNG page, a ``.txt`` file holding the
exact rendered text, and a ``.json`` file of the structured fields. The JSON is
not used by the current OCR evaluation; it is the supervision signal for the
field-extraction work described in ``docs/EXTENDING.md``.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont
from tqdm import tqdm

from doculens.config import Config, DataConfig

logger = logging.getLogger(__name__)

VALUE_COLUMN_OFFSET = 440
"""Horizontal offset of the value column from the left margin, in pixels."""

_STATE_CODES = ("MH", "KA", "DL", "TN", "GJ", "UP", "RJ", "WB", "TS", "KL", "HR", "PB")
_VEHICLE_MAKES = (
    ("Maruti Suzuki", ("Swift", "Baleno", "Dzire", "Brezza", "Alto K10")),
    ("Hyundai", ("i20", "Creta", "Venue", "Verna", "Grand i10")),
    ("Tata Motors", ("Nexon", "Punch", "Harrier", "Altroz", "Tiago")),
    ("Mahindra", ("Scorpio-N", "XUV700", "Thar", "Bolero", "XUV300")),
    ("Honda", ("City", "Amaze", "Elevate", "Jazz", "WR-V")),
    ("Toyota", ("Innova Crysta", "Fortuner", "Glanza", "Urban Cruiser", "Etios")),
)
_LOSS_NATURES = (
    "Front bumper and headlamp damage",
    "Rear-end collision damage",
    "Side door dent and panel scratch",
    "Windscreen crack",
    "Water ingress damage to engine bay",
    "Theft of accessories",
    "Hail and falling object damage",
)
_INSURERS = (
    "BHARAT GENERAL INSURANCE CO. LTD.",
    "SANGAM GENERAL ASSURANCE LTD.",
    "DECCAN SHIELD INSURANCE CO. LTD.",
)
_GARAGES = (
    "SHREE AUTO WORKS",
    "NEW BHARAT MOTORS SERVICE",
    "PRECISION CAR CARE CENTRE",
    "SAI MULTIBRAND GARAGE",
)
_PAYMENT_MODES = ("NEFT", "RTGS", "UPI", "Cheque", "Cashless Settlement")
_SURVEY_STATUSES = ("Surveyed - Approved", "Surveyed - Partially Approved", "Pending Survey")

_GST_RATE_PERCENT = 9.0
_DOC_ID_WIDTH = 3


@dataclass(frozen=True)
class DocLine:
    """One rendered line of a synthetic document.

    Attributes:
        kind: ``title``, ``section``, ``field`` or ``blank``.
        text: Line text for ``title`` and ``section`` kinds.
        label: Field label, without the trailing colon, for ``field`` kinds.
        value: Field value for ``field`` kinds.
        key: ``snake_case`` key under which a ``field`` is stored in the JSON.
    """

    kind: str
    text: str = ""
    label: str = ""
    value: str = ""
    key: str = ""

    def as_text(self) -> str:
        """Return this line as it appears in the ground-truth text."""
        if self.kind == "field":
            return f"{self.label}: {self.value}"
        return self.text


@dataclass(frozen=True)
class GeneratedDocument:
    """A synthetic document together with its ground truth."""

    doc_id: str
    doc_type: str
    image: np.ndarray
    """The rendered page as a 2-D ``uint8`` grayscale array."""
    text: str
    """The exact rendered text: non-blank lines joined by newlines."""
    fields: dict[str, str] = field(default_factory=dict)
    """Structured field values, for future field-extraction supervision."""


def _title(text: str) -> DocLine:
    """Return a title line."""
    return DocLine(kind="title", text=text)


def _section(text: str) -> DocLine:
    """Return a section-heading line."""
    return DocLine(kind="section", text=text)


def _field(key: str, label: str, value: str) -> DocLine:
    """Return a label/value field line."""
    return DocLine(kind="field", label=label, value=value, key=key)


def _blank() -> DocLine:
    """Return a vertical spacer line."""
    return DocLine(kind="blank")


def format_inr(amount: float) -> str:
    """Format a rupee amount with Indian digit grouping.

    Indian grouping puts the last three digits together and then groups in
    twos, so 123456.5 becomes ``1,23,456.50``.

    Args:
        amount: Amount in rupees.

    Returns:
        A string of the form ``INR 1,23,456.50``.
    """
    rupees, paise = divmod(round(float(amount) * 100), 100)
    digits = str(int(rupees))
    if len(digits) > 3:
        head, tail = digits[:-3], digits[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        digits = ",".join([*groups, tail])
    return f"INR {digits}.{paise:02d}"


def _vehicle_registration(rng: random.Random) -> str:
    """Return a plausible Indian vehicle registration number.

    Args:
        rng: Seeded random source.

    Returns:
        A registration of the form ``MH 12 AB 3456``.
    """
    letters = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    state = rng.choice(_STATE_CODES)
    district = rng.randint(1, 48)
    series = "".join(rng.choice(letters) for _ in range(2))
    number = rng.randint(1000, 9999)
    return f"{state} {district:02d} {series} {number}"


def _phone(rng: random.Random) -> str:
    """Return a plausible Indian mobile number in +91 format."""
    return f"+91 {rng.randint(60, 99)}{rng.randint(10000000, 99999999)}"


def _policy_number(rng: random.Random) -> str:
    """Return a plausible policy number such as ``POL-MH-2023-004821``."""
    return f"POL-{rng.choice(_STATE_CODES)}-{rng.randint(2019, 2025)}-{rng.randint(1, 999999):06d}"


def _claim_number(rng: random.Random) -> str:
    """Return a plausible claim number such as ``CLM/2024/0481234``."""
    return f"CLM/{rng.randint(2022, 2025)}/{rng.randint(1, 9999999):07d}"


def _gstin(rng: random.Random) -> str:
    """Return a syntactically plausible GSTIN."""
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    pan = (
        "".join(rng.choice(letters) for _ in range(5))
        + f"{rng.randint(0, 9999):04d}"
        + rng.choice(letters)
    )
    return f"{rng.randint(1, 37):02d}{pan}{rng.randint(1, 9)}Z{rng.choice(letters)}"


def _claim_form_lines(fake: Faker, rng: random.Random) -> list[DocLine]:
    """Build the line layout of a motor-insurance claim form.

    Args:
        fake: Seeded Faker instance using the configured locale.
        rng: Seeded random source for choices not covered by Faker.

    Returns:
        The ordered lines of the document.
    """
    make, models = rng.choice(_VEHICLE_MAKES)
    estimated = rng.randint(8_000, 480_000) + rng.randint(0, 99) / 100
    claimed = estimated * rng.uniform(0.72, 0.99)
    accident_date = fake.date_between(start_date="-2y", end_date="-10d")
    return [
        _title(rng.choice(_INSURERS)),
        _title("MOTOR INSURANCE CLAIM FORM"),
        _blank(),
        _section("POLICY AND CLAIM DETAILS"),
        _field("policy_number", "Policy Number", _policy_number(rng)),
        _field("claim_number", "Claim Number", _claim_number(rng)),
        _field(
            "claim_date",
            "Date of Intimation",
            fake.date_between(start_date=accident_date, end_date="today").strftime("%d/%m/%Y"),
        ),
        _field(
            "policy_type",
            "Policy Type",
            rng.choice(
                (
                    "Private Car Comprehensive",
                    "Private Car Own Damage",
                    "Commercial Vehicle Package",
                )
            ),
        ),
        _blank(),
        _section("INSURED PERSON"),
        _field("insured_name", "Policyholder Name", fake.name()),
        _field("contact_number", "Contact Number", _phone(rng)),
        _field("email", "Email Address", fake.ascii_email()),
        _field("address_line", "Address", fake.street_address().replace("\n", ", ")),
        _field("city", "City", fake.city()),
        _field("state", "State", fake.state()),
        _field("pincode", "PIN Code", fake.postcode()),
        _blank(),
        _section("VEHICLE DETAILS"),
        _field("vehicle_registration", "Registration Number", _vehicle_registration(rng)),
        _field("vehicle_make", "Make", make),
        _field("vehicle_model", "Model", rng.choice(models)),
        _field("manufacture_year", "Year of Manufacture", str(rng.randint(2012, 2025))),
        _field(
            "engine_number",
            "Engine Number",
            f"{rng.randint(10000000, 99999999)}{rng.choice('ABCDEFGH')}",
        ),
        _blank(),
        _section("LOSS DETAILS"),
        _field("accident_date", "Date of Accident", accident_date.strftime("%d/%m/%Y")),
        _field("accident_place", "Place of Accident", f"{fake.city()}, {fake.state()}"),
        _field("nature_of_loss", "Nature of Loss", rng.choice(_LOSS_NATURES)),
        _field("estimated_loss", "Estimated Loss", format_inr(estimated)),
        _field("claim_amount", "Amount Claimed", format_inr(claimed)),
        _field("survey_status", "Survey Status", rng.choice(_SURVEY_STATUSES)),
        _field("surveyor_name", "Surveyor Name", fake.name()),
        _blank(),
        _section("DECLARATION"),
        _field("declaration_place", "Place", fake.city()),
        _field(
            "declaration_date",
            "Date",
            fake.date_between(start_date=accident_date, end_date="today").strftime("%d/%m/%Y"),
        ),
        _field("declaration_signatory", "Signature of Insured", fake.name()),
    ]


def _invoice_lines(fake: Faker, rng: random.Random) -> list[DocLine]:
    """Build the line layout of a garage repair invoice.

    Monetary fields are internally consistent: subtotal is the sum of labour
    and parts, GST is charged at the configured rate on each of CGST and SGST,
    and the total is their sum.

    Args:
        fake: Seeded Faker instance using the configured locale.
        rng: Seeded random source for choices not covered by Faker.

    Returns:
        The ordered lines of the document.
    """
    make, models = rng.choice(_VEHICLE_MAKES)
    labour = rng.randint(1_500, 42_000) + rng.randint(0, 99) / 100
    parts = rng.randint(2_000, 310_000) + rng.randint(0, 99) / 100
    subtotal = labour + parts
    cgst = subtotal * _GST_RATE_PERCENT / 100.0
    total = subtotal + 2 * cgst
    invoice_date = fake.date_between(start_date="-1y", end_date="today")
    return [
        _title(rng.choice(_GARAGES)),
        _title("TAX INVOICE - ACCIDENT REPAIR"),
        _blank(),
        _section("INVOICE DETAILS"),
        _field(
            "invoice_number",
            "Invoice Number",
            f"INV/{invoice_date.year}/{rng.randint(1, 99999):05d}",
        ),
        _field("invoice_date", "Invoice Date", invoice_date.strftime("%d/%m/%Y")),
        _field("gstin", "GSTIN", _gstin(rng)),
        _field("workshop_address", "Workshop Address", fake.street_address().replace("\n", ", ")),
        _field("workshop_city", "City", fake.city()),
        _blank(),
        _section("CUSTOMER AND POLICY"),
        _field("customer_name", "Customer Name", fake.name()),
        _field("contact_number", "Contact Number", _phone(rng)),
        _field("policy_number", "Policy Number", _policy_number(rng)),
        _field("claim_number", "Claim Number", _claim_number(rng)),
        _field("insurer_name", "Insurer", rng.choice(_INSURERS)),
        _blank(),
        _section("VEHICLE"),
        _field("vehicle_registration", "Registration Number", _vehicle_registration(rng)),
        _field("vehicle_make_model", "Make and Model", f"{make} {rng.choice(models)}"),
        _field("odometer_km", "Odometer Reading", f"{rng.randint(1200, 189000)} km"),
        _blank(),
        _section("CHARGES"),
        _field("labour_charges", "Labour Charges", format_inr(labour)),
        _field("parts_cost", "Spare Parts Cost", format_inr(parts)),
        _field("subtotal", "Subtotal", format_inr(subtotal)),
        _field("cgst", f"CGST @ {_GST_RATE_PERCENT:.0f}%", format_inr(cgst)),
        _field("sgst", f"SGST @ {_GST_RATE_PERCENT:.0f}%", format_inr(cgst)),
        _field("total_amount", "Total Amount Payable", format_inr(total)),
        _field("payment_mode", "Payment Mode", rng.choice(_PAYMENT_MODES)),
        _field("authorised_by", "Authorised By", fake.name()),
        _blank(),
        _section("TERMS"),
        _field("warranty_months", "Parts Warranty", f"{rng.choice((3, 6, 12, 24))} months"),
        _field(
            "delivery_date",
            "Vehicle Delivery Date",
            fake.date_between(start_date=invoice_date, end_date="+30d").strftime("%d/%m/%Y"),
        ),
        _field("service_advisor", "Service Advisor", fake.name()),
    ]


_LAYOUTS = {"claim_form": _claim_form_lines, "invoice": _invoice_lines}


def available_doc_types() -> list[str]:
    """Return the sorted names of the implemented document layouts."""
    return sorted(_LAYOUTS)


def resolve_font(candidates: tuple[str, ...], size: int) -> ImageFont.ImageFont:
    """Load the first available TrueType font, with graceful fallbacks.

    The configured candidate paths are tried in order. If none exists the
    DejaVu Sans shipped with matplotlib is used, which is always present
    because matplotlib is a dependency. Only if that also fails does this fall
    back to PIL's small bitmap font, which is logged as a warning because it
    makes pages look nothing like a scan.

    Args:
        candidates: Candidate font file paths from the config.
        size: Requested point size.

    Returns:
        A PIL font object.
    """
    for candidate in candidates:
        path = Path(candidate).expanduser()
        if path.is_file():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                logger.debug("font %s exists but could not be loaded", path)
    try:
        import matplotlib

        bundled = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
        if bundled.is_file():
            logger.debug("using matplotlib-bundled font %s", bundled)
            return ImageFont.truetype(str(bundled), size=size)
    except (ImportError, OSError):
        logger.debug("matplotlib-bundled DejaVu Sans unavailable")
    logger.warning(
        "no TrueType font found (tried %s); falling back to PIL's bitmap default, "
        "which will not look like a scanned page",
        list(candidates),
    )
    return ImageFont.load_default()


def render_document(lines: list[DocLine], data: DataConfig) -> np.ndarray:
    """Render document lines onto a blank page.

    Titles are centred and drawn bold, section headings are drawn bold at the
    left margin, and fields are drawn as a bold label at the margin with the
    value in a second column. Collapsing whitespace during evaluation makes
    that two-column gap equivalent to a single space.

    Args:
        lines: Ordered lines to render.
        data: Data configuration supplying canvas, margin, spacing and fonts.

    Returns:
        The page as a 2-D ``uint8`` grayscale array.
    """
    canvas = PILImage.new(
        "L", (data.canvas.width, data.canvas.height), color=data.canvas.background
    )
    draw = ImageDraw.Draw(canvas)
    body = resolve_font(data.font.candidates, data.font.size_body)
    body_bold = resolve_font(data.font.bold_candidates, data.font.size_body)
    title_bold = resolve_font(data.font.bold_candidates, data.font.size_title)

    y = data.margin
    value_x = data.margin + VALUE_COLUMN_OFFSET
    for line in lines:
        if line.kind == "blank":
            y += data.line_spacing
            continue
        if line.kind == "title":
            width = draw.textlength(line.text, font=title_bold)
            draw.text(
                ((data.canvas.width - width) / 2.0, y),
                line.text,
                font=title_bold,
                fill=data.text_colour,
            )
            y += data.line_spacing + (data.font.size_title - data.font.size_body)
        elif line.kind == "section":
            draw.text((data.margin, y), line.text, font=body_bold, fill=data.text_colour)
            y += data.line_spacing
        else:
            draw.text((data.margin, y), f"{line.label}:", font=body_bold, fill=data.text_colour)
            draw.text((value_x, y), line.value, font=body, fill=data.text_colour)
            y += data.line_spacing
    if y > data.canvas.height:
        logger.warning(
            "document content ran %d px past the bottom of the %dpx canvas; "
            "reduce data.line_spacing or shorten the layout",
            y - data.canvas.height,
            data.canvas.height,
        )
    return np.array(canvas, dtype=np.uint8)


def ground_truth_text(lines: list[DocLine]) -> str:
    """Return the exact text of a document: non-blank lines joined by newlines.

    Args:
        lines: Ordered lines of the document.

    Returns:
        The ground-truth transcription.
    """
    return "\n".join(line.as_text() for line in lines if line.kind != "blank")


def structured_fields(lines: list[DocLine]) -> dict[str, str]:
    """Extract the ``key -> value`` mapping of the document's field lines.

    Args:
        lines: Ordered lines of the document.

    Returns:
        The structured field values.
    """
    return {line.key: line.value for line in lines if line.kind == "field" and line.key}


def generate_document(doc_id: str, doc_type: str, seed: int, data: DataConfig) -> GeneratedDocument:
    """Generate one synthetic document and its ground truth.

    The Faker instance and the random source are seeded per document from
    ``seed``, so a document's content depends only on its own seed and never on
    how many documents were generated before it.

    Args:
        doc_id: Identifier such as ``doc_007``.
        doc_type: One of :func:`available_doc_types`.
        seed: Per-document seed.
        data: Data configuration.

    Returns:
        The generated document.

    Raises:
        ValueError: If ``doc_type`` has no implemented layout.
    """
    if doc_type not in _LAYOUTS:
        raise ValueError(
            f"unknown doc_type {doc_type!r}; implemented layouts are {available_doc_types()}"
        )
    fake = Faker(data.locale)
    fake.seed_instance(seed)
    rng = random.Random(seed)
    lines = _LAYOUTS[doc_type](fake, rng)
    return GeneratedDocument(
        doc_id=doc_id,
        doc_type=doc_type,
        image=render_document(lines, data),
        text=ground_truth_text(lines),
        fields=structured_fields(lines),
    )


def assign_splits(doc_ids: list[str], data: DataConfig, seed: int) -> dict[str, str]:
    """Assign documents to train/val/test deterministically.

    Splits are fixed here, before any pipeline is run or tuned, and recorded in
    the manifest. Only the test split is used for reported numbers.

    Args:
        doc_ids: All document identifiers.
        data: Data configuration holding the split fractions.
        seed: Seed for the shuffle.

    Returns:
        A mapping of document id to ``train``, ``val`` or ``test``.
    """
    shuffled = list(doc_ids)
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    n_train = int(round(data.split.train * total))
    n_val = int(round(data.split.val * total))
    n_train = min(n_train, total)
    n_val = min(n_val, total - n_train)
    assignment: dict[str, str] = {}
    for index, doc_id in enumerate(shuffled):
        if index < n_train:
            assignment[doc_id] = "train"
        elif index < n_train + n_val:
            assignment[doc_id] = "val"
        else:
            assignment[doc_id] = "test"
    return assignment


def generate_corpus(
    config: Config, n_docs: int | None = None, show_progress: bool = True
) -> pd.DataFrame:
    """Generate the clean corpus, write it to disk and return its manifest.

    For each document three files are written to ``paths.clean_dir``:
    ``<doc_id>.png``, ``<doc_id>.txt`` (exact ground truth) and
    ``<doc_id>.json`` (structured fields). The manifest is written to
    ``paths.manifest_clean``.

    Args:
        config: Validated configuration.
        n_docs: Override for ``data.n_docs``, used by the quick smoke runs.
        show_progress: Whether to display a progress bar.

    Returns:
        The manifest as a DataFrame with one row per document.
    """
    config.paths.ensure_dirs()
    count = int(n_docs if n_docs is not None else config.data.n_docs)
    if count < 1:
        raise ValueError(f"n_docs must be >= 1, got {count}")
    doc_types = list(config.data.doc_types)
    doc_ids = [f"doc_{index + 1:0{_DOC_ID_WIDTH}d}" for index in range(count)]
    splits = assign_splits(doc_ids, config.data, config.seed)

    rows: list[dict[str, object]] = []
    iterator = enumerate(doc_ids)
    if show_progress:
        iterator = tqdm(list(iterator), desc="generating documents", unit="doc")
    for index, doc_id in iterator:
        doc_type = doc_types[index % len(doc_types)]
        document = generate_document(
            doc_id=doc_id,
            doc_type=doc_type,
            seed=config.seed + index,
            data=config.data,
        )
        png_path = config.paths.clean_dir / f"{doc_id}.png"
        txt_path = config.paths.clean_dir / f"{doc_id}.txt"
        json_path = config.paths.clean_dir / f"{doc_id}.json"
        PILImage.fromarray(document.image).save(png_path)
        txt_path.write_text(document.text, encoding="utf-8")
        json_path.write_text(
            json.dumps(
                {"doc_id": doc_id, "doc_type": doc_type, "fields": document.fields},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        rows.append(
            {
                "doc_id": doc_id,
                "doc_type": doc_type,
                "split": splits[doc_id],
                "png": str(png_path.relative_to(config.paths.root)),
                "txt": str(txt_path.relative_to(config.paths.root)),
                "json": str(json_path.relative_to(config.paths.root)),
                "n_chars": len(document.text),
                "n_words": len(document.text.split()),
                "n_fields": len(document.fields),
            }
        )

    manifest = pd.DataFrame(rows)
    manifest.to_csv(config.paths.manifest_clean, index=False)
    logger.info(
        "generated %d documents (%s) -> %s",
        len(manifest),
        ", ".join(f"{k}={v}" for k, v in manifest["split"].value_counts().sort_index().items()),
        config.paths.clean_dir,
    )
    return manifest
