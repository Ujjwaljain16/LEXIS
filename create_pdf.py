from fpdf import FPDF

pdf = FPDF()
pdf.add_page()
pdf.set_font("Helvetica", size=12)
pdf.multi_cell(0, 10, text="""
NON-DISCLOSURE AGREEMENT

This Non-Disclosure Agreement (the "Agreement") is entered into this 1st day of June 2026, by and between Lexis Corp ("Disclosing Party") and John Doe ("Receiving Party").

1. Confidential Information. "Confidential Information" means any and all technical and non-technical information provided by Disclosing Party to Receiving Party, which may include without limitation information regarding: (a) patent and patent applications; (b) trade secrets; and (c) proprietary and confidential information.

2. Obligations of Receiving Party. Receiving Party agrees that it shall take reasonable measures to protect the secrecy of and avoid disclosure and unauthorized use of the Confidential Information of Disclosing Party. Without limiting the foregoing, Receiving Party shall take at least those measures that it takes to protect its own most highly confidential information.

3. No License. Nothing in this Agreement is intended to grant any rights to Receiving Party under any patent, mask work right or copyright of Disclosing Party, nor shall this Agreement grant Receiving Party any rights in or to the Confidential Information except as expressly set forth herein.

4. Term. The obligations of Receiving Party under this Agreement shall survive until such time as all Confidential Information of Disclosing Party disclosed hereunder becomes publicly known and made generally available through no action or inaction of Receiving Party.

5. Miscellaneous. This Agreement shall bind and inure to the benefit of the parties hereto and their successors and assigns. This Agreement shall be governed by the laws of the State of California, without reference to conflict of laws principles.
""")

pdf.output("sample_contract.pdf")
