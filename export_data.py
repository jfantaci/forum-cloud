"""
Export word_freq_by_year to JavaScript format.

Run this after generating your word_freq_by_year dictionary.
Copy the output and paste it into the DATA constant
in fletcher-word-cloud.html.
"""

import json


def export_for_html(word_freq_by_year):
    """
    Convert word_freq_by_year to JavaScript object string.
    Sorts words by frequency (descending) within each year.
    """
    output = {}
    for year in sorted(word_freq_by_year.keys()):
        freqs = word_freq_by_year[year]
        # Sort by frequency descending
        sorted_freqs = dict(sorted(freqs.items(), key=lambda x: -x[1]))
        output[str(year)] = sorted_freqs

    js_string = json.dumps(output, indent=2, ensure_ascii=False)

    print("=" * 60)
    print("Copy everything between the lines into your HTML file,")
    print("replacing the existing DATA constant:")
    print("=" * 60)
    print()
    print(f"const DATA = {js_string};")
    print()
    print("=" * 60)

    # Also save to a file
    with open("word_cloud_data.js", "w", encoding="utf-8") as f:
        f.write(f"const DATA = {js_string};")

    print("Also saved to: word_cloud_data.js")


# ============================================================
# USAGE: Call this after your processing code
# ============================================================
# export_for_html(word_freq_by_year)