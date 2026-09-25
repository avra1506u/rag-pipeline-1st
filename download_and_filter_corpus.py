"""
Downloads a LARGE batch of arXiv PDFs from the Vectara Open RAG Benchmark,
scores each by text density (chars per page), and keeps only the top N
most prose-heavy ones for the RAG corpus.

Papers that are table/figure-heavy get a low text-density score and are
filtered out automatically - no manual eyeballing needed.

Usage:
    1. pip install pymupdf --break-system-packages
    2. Edit BATCH_SIZE and KEEP_TOP_N below if you want to adjust.
    3. Run: python download_and_filter_corpus.py
    4. Check data/pdf/ for the kept papers, and corpus_report.txt for scores.
"""

import os
import time
import json
import urllib.request
import fitz  # PyMuPDF

# --- Config ---
BATCH_SIZE = 200      # how many candidate papers to download and score
KEEP_TOP_N = 20        # how many of the best (most text-dense) to actually keep
OUTPUT_DIR = "data/pdf"
STAGING_DIR = "data/pdf_candidates"  # temp folder for the full batch before filtering

# Paste the arXiv IDs you want to sample from here (e.g. a slice of
# pdf_urls.json, or IDs you've picked by category). Using a batch here as
# an example - replace with your own selection.
CANDIDATE_IDS = [
    "2407.01528v3", "2412.00651v1", "2411.00713v1", "2411.16245v2",
    "2402.03953v4", "2408.08383v2", "2410.19011v2", "2411.03915v2",
    "2404.18884v2", "2410.08709v3", "2409.19777v2", "2405.01155v3",
    "2405.02054v2", "2409.16507v2", "2412.18222v1", "2411.12363v3",
    "2411.18627v2", "2403.18177v2", "2412.08070v2", "2409.13467v3",
    "2407.02634v3", "2407.11336v2", "2411.15638v2", "2402.12350v3",
    "2406.04991v2", "2408.16891v2", "2410.00158v1", "2412.14784v3",
    "2412.01100v2", "2412.12639v3", "2410.20081v3", "2412.17119v3",
    "2409.14494v3", "2411.02558v1", "2406.18046v2", "2411.07216v2",
    "2407.11758v1", "2402.05967v7", "2412.12919v2", "2410.00908v2",
    "2406.01756v2", "2403.15076v1", "2406.18345v3", "2409.20319v2",
    "2412.04468v2", "2407.18909v1", "2407.03285v2", "2412.06412v2",
    "2407.04378v2", "2412.15576v4", "2408.02322v2", "2406.09018v2",
    "2412.16874v3", "2410.11034v2", "2406.10612v1", "2412.11037v2",
    "2410.10474v1", "2404.17736v3", "2412.05430v1", "2408.09648v4",
    "2410.20982v2", "2411.08072v1", "2405.18213v3", "2412.16067v1",
    "2407.07368v3", "2408.07379v2", "2404.18869v2", "2403.13637v6",
    "2412.17780v3", "2410.03930v3", "2412.20019v2", "2407.11894v2",
    "2412.00886v2", "2411.14471v1", "2412.00566v2", "2411.01864v1",
    "2408.14872v2", "2412.14029v2", "2411.02807v4", "2412.20245v4",
    "2401.08028v3", "2411.18473v2", "2406.06650v2", "2402.16901v2",
    "2501.00225v2", "2405.05881v2", "2404.13877v2", "2410.23473v2",
    "2410.04943v2", "2411.12193v2", "2412.18252v2", "2405.06851v2",
    "2412.04313v2", "2405.12292v3", "2412.04997v2", "2410.16441v2",
    "2402.07135v2", "2408.06427v2", "2409.12516v1", "2403.03363v6",
    "2404.06803v2", "2408.08408v3", "2405.20415v3", "2405.03910v2",
    "2408.07618v3", "2403.11738v3", "2410.10304v2", "2409.06325v3",
    "2408.09344v2", "2412.16352v2", "2407.09976v3", "2403.01421v2",
    "2410.01890v2", "2402.03637v2", "2406.05031v2", "2401.14085v2",
    "2411.13673v2", "2403.06613v3", "2411.03988v2", "2409.17606v2",
    "2407.08797v3", "2401.02247v4", "2409.00578v2", "2404.18775v4",
    "2407.10577v1", "2410.19236v2", "2408.04814v3", "2410.19599v3",
    "2406.14184v3", "2406.07366v2", "2412.10243v3", "2411.02951v2",
    "2408.10999v2", "2406.12070v3", "2412.15322v2", "2410.05459v2",
    "2408.16309v2", "2411.13783v2", "2412.09374v3", "2410.02927v1",
    "2405.19529v4", "2409.02476v2", "2410.24191v1", "2410.05401v2",
    "2411.16277v1", "2408.11878v2", "2412.18432v3", "2501.00058v1",
    "2411.18018v2", "2412.01649v1", "2411.11059v1", "2412.07587v6",
    "2401.02564v2", "2412.09297v2", "2410.22568v1", "2407.11761v3",
    "2411.19444v3", "2412.20317v3", "2407.13130v2", "2410.23587v3",
    "2406.09670v4", "2404.17763v2", "2404.07575v4", "2402.01892v2",
    "2412.14639v2", "2409.14585v2", "2407.02022v2", "2406.17567v1",
    "2411.05391v4", "2404.08757v2", "2407.08192v3", "2404.11929v3",
    "2404.16880v3", "2407.09711v4", "2406.13580v2", "2402.10287v2",
    "2409.17635v2", "2411.01983v2", "2402.04703v2", "2410.01265v2",
    "2412.14369v2", "2406.08366v2", "2412.06611v2", "2411.02729v2",
    "2409.13674v3", "2405.16924v2", "2407.16566v5", "2411.07984v2",
    "2406.16246v4", "2410.06580v3", "2411.10728v3", "2412.09393v2",
    "2411.13384v2", "2409.15621v2", "2406.10332v2", "2408.13230v2",
    "2411.03676v2", "2412.02459v2", "2412.13914v3", "2402.17413v3",
][:BATCH_SIZE]


def download(arxiv_id: str, output_dir: str) -> str | None:
    url = f"https://arxiv.org/pdf/{arxiv_id}"
    filename = f"{arxiv_id}.pdf"
    filepath = os.path.join(output_dir, filename)

    if os.path.exists(filepath):
        return filepath

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as response:
            data = response.read()
        with open(filepath, "wb") as f:
            f.write(data)
        return filepath
    except Exception as e:
        print(f"  FAILED download {filename}: {e}")
        return None


def score_text_density(filepath: str) -> dict:
    """Returns chars-per-page - a rough proxy for how prose-heavy a PDF is.
    Low score = likely table/figure/formula heavy. High score = prose-dense."""
    try:
        doc = fitz.open(filepath)
        num_pages = len(doc)
        total_chars = 0
        for page in doc:
            total_chars += len(page.get_text())
        doc.close()
        if num_pages == 0:
            return {"pages": 0, "chars": 0, "density": 0}
        return {
            "pages": num_pages,
            "chars": total_chars,
            "density": round(total_chars / num_pages, 1),
        }
    except Exception as e:
        return {"pages": 0, "chars": 0, "density": 0, "error": str(e)}


def main():
    os.makedirs(STAGING_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Downloading {len(CANDIDATE_IDS)} candidate papers...\n")
    results = []
    for arxiv_id in CANDIDATE_IDS:
        filepath = download(arxiv_id, STAGING_DIR)
        if filepath is None:
            continue
        score = score_text_density(filepath)
        score["id"] = arxiv_id
        score["filepath"] = filepath
        results.append(score)
        print(f"  {arxiv_id}: {score['pages']} pages, density={score['density']} chars/page")
        time.sleep(1)  # be polite to arXiv

    # Sort by density descending, keep top N
    results = [r for r in results if r["density"] > 0]
    results.sort(key=lambda r: r["density"], reverse=True)
    kept = results[:KEEP_TOP_N]
    dropped = results[KEEP_TOP_N:]

    print(f"\n--- Keeping top {len(kept)} most text-dense papers ---")
    for r in kept:
        dest = os.path.join(OUTPUT_DIR, os.path.basename(r["filepath"]))
        os.replace(r["filepath"], dest)
        print(f"  KEPT {r['id']} (density={r['density']})")

    print(f"\n--- Dropped {len(dropped)} lower-density (likely table/figure-heavy) papers ---")
    for r in dropped:
        print(f"  dropped {r['id']} (density={r['density']})")

    # Write a report
    with open("corpus_report.txt", "w") as f:
        f.write("Text-density corpus filtering report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Kept {len(kept)} of {len(results)} downloaded papers\n\n")
        f.write("KEPT:\n")
        for r in kept:
            f.write(f"  {r['id']}: {r['pages']} pages, {r['density']} chars/page\n")
        f.write("\nDROPPED:\n")
        for r in dropped:
            f.write(f"  {r['id']}: {r['pages']} pages, {r['density']} chars/page\n")

    print(f"\nDone. See corpus_report.txt for full details.")
    print(f"Kept papers are in {OUTPUT_DIR}/, ready for your ingestion notebook.")
    print(f"You may want to delete {STAGING_DIR}/ now (unused dropped PDFs still there).")


if __name__ == "__main__":
    main()