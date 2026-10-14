"""Builds eval/handbook.pdf: a fictional 5-page employee handbook used to test DocMind.

Page 4 is a scanned image with no text layer, so the OCR path is tested too.
Every question in questions.json is answerable from exactly one place here,
or deliberately not answerable at all.
"""
from pathlib import Path

PAGES = [
    ("Northwind Robotics: Employee Handbook", [
        "Northwind Robotics builds warehouse picking robots. The company was founded in 2017 by Asha Raman and Tomas Berg in Pune, India.",
        "Today Northwind has 340 employees across three offices: Pune, Rotterdam and Austin. The headquarters is in Pune.",
        "This handbook explains how we work, what we offer, and what we expect. It applies to all full-time employees. Contractors follow the terms in their contracts.",
        "If anything here is unclear, ask People Operations at people@northwind.example.",
    ]),
    ("Working hours and leave", [
        "Core hours are 11:00 to 16:00 local time. Outside core hours, employees choose when they work, as long as they work 40 hours a week.",
        "Employees get 24 days of paid annual leave per year, plus public holidays of the office they belong to. Up to 5 unused leave days can be carried over to the next year.",
        "Sick leave is 12 days per year. A doctor's note is needed only for absences longer than 3 consecutive days.",
        "Parental leave is 26 weeks at full pay for the primary caregiver and 8 weeks at full pay for the secondary caregiver.",
        "Every employee may work remotely for up to 2 days per week. Fully remote arrangements need approval from a director.",
    ]),
    ("Pay, equipment and learning", [
        "Salaries are paid on the last working day of each month. Salary reviews happen once a year, in April.",
        "New employees receive a laptop of their choice from an approved list, and a one-time home office allowance of 600 euros.",
        "Each employee has an annual learning budget of 1,500 euros for courses, books and conferences. Unused budget does not roll over.",
        "Employees who refer a candidate who is hired receive a referral bonus of 2,000 euros, paid after the new hire completes 6 months.",
    ]),
    ("Safety on the warehouse floor", [
        "Anyone entering a test warehouse must wear safety shoes and a high-visibility vest. Hard hats are required only in the Rotterdam high-bay area.",
        "Robots in test mode are limited to a speed of 1.2 metres per second when people are within 5 metres.",
        "Every robot has a red emergency stop button on both sides. Any employee may press it at any time, without asking.",
        "Incidents must be reported in the SafeLog system within 24 hours, even if nobody was hurt.",
        "New hardware engineers complete a 2-day safety training before working on the floor.",
    ]),
    ("Travel and expenses", [
        "Business travel must be approved by your manager before booking. Flights under 6 hours are booked in economy class.",
        "Hotels are booked through the company travel portal. The nightly limit is 150 euros in Europe and 180 dollars in the United States.",
        "Meals during business travel are reimbursed up to 60 euros per day. Alcohol is not reimbursed.",
        "Expense claims must be submitted within 30 days, with a photo of every receipt.",
        "Train travel is preferred over flights for trips under 4 hours.",
    ]),
    ("Performance and growth", [
        "Every employee has a one-to-one meeting with their manager every two weeks.",
        "Performance reviews happen twice a year, in March and September, and include feedback from at least two peers.",
        "Engineers follow a career ladder with six levels, from Engineer I to Principal Engineer. Promotions are decided by a committee, not by a single manager.",
        "Employees can move to another team after 12 months in their current role, if the new team has an open position.",
        "Mentoring is voluntary. People Operations matches mentors and mentees each quarter.",
    ]),
    ("Onboarding", [
        "On the first day, new employees meet their buddy, a colleague from another team who answers practical questions during the first month.",
        "The first week includes a product demo, a visit to the test warehouse and a session on company history.",
        "Every new employee writes a 30-60-90 day plan with their manager in the first two weeks.",
        "Access to production systems is granted only after the security training is completed.",
    ]),
    ("Communication", [
        "Most discussion happens in written form in the team channels, so people in different time zones can follow it.",
        "Meetings should have an agenda shared at least one day in advance. Meetings without an agenda can be declined.",
        "The whole company meets once a month for an all-hands meeting, which is recorded for those who cannot attend.",
        "Urgent production issues are raised in the incident channel, not by direct message.",
    ]),
    ("Office life", [
        "Each office has a kitchen with free coffee, tea and fruit. Lunch is provided on Wednesdays.",
        "Desks are not assigned. Employees book a desk through the office app the day before.",
        "Dogs are allowed in the Rotterdam office on Fridays. Other offices do not allow pets.",
        "The office opens at 7:00 and closes at 21:00. Access outside these hours needs approval from facilities.",
    ]),
    ("Security and conduct", [
        "Two-factor authentication is mandatory for all company accounts. Passwords must be at least 14 characters long.",
        "Customer warehouse data may only be stored in the company's Frankfurt data region.",
        "Company devices must be locked when unattended. Lost devices must be reported to IT within 1 hour.",
        "Gifts from suppliers worth more than 50 euros must be declared to the compliance team.",
        "Violations of this handbook are handled by People Operations and may lead to a written warning or termination.",
    ]),
]
SCANNED_PAGE = 3  # 0-based: page 4 is an image only


def build(out: Path):
    from io import BytesIO

    from PIL import Image, ImageDraw, ImageFont
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas
    from reportlab.platypus import Frame, Paragraph

    styles = getSampleStyleSheet()
    c = canvas.Canvas(str(out), pagesize=A4)
    w, h = A4
    for n, (title, paras) in enumerate(PAGES):
        if n == SCANNED_PAGE:
            # Render the page as a picture, like a scan: no text layer at all.
            img = Image.new("L", (1240, 1754), 255)
            d = ImageDraw.Draw(img)
            try:
                big = ImageFont.truetype("DejaVuSans-Bold.ttf", 40)
                font = ImageFont.truetype("DejaVuSans.ttf", 28)
            except OSError:
                big = font = ImageFont.load_default()
            y = 140
            d.text((110, y), title, font=big, fill=0)
            y += 90
            for p in paras:
                line = ""
                for word in p.split():
                    if d.textlength(line + " " + word, font=font) > 1020:
                        d.text((110, y), line.strip(), font=font, fill=0)
                        y += 42
                        line = ""
                    line += " " + word
                d.text((110, y), line.strip(), font=font, fill=0)
                y += 70
            buf = BytesIO()
            img.save(buf, "PNG")
            buf.seek(0)
            c.drawImage(ImageReader(buf), 0, 0, w, h)
        else:
            story = [Paragraph(title, styles["Title"])] + [Paragraph(p, styles["BodyText"]) for p in paras]
            Frame(60, 60, w - 120, h - 120).addFromList(story, c)
        c.showPage()
    c.save()


if __name__ == "__main__":
    build(Path(__file__).with_name("handbook.pdf"))
    print("wrote eval/handbook.pdf")
