import os
import io
import json
import re
import warnings
import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk
import threading
 
import numpy as np
import pdfplumber
import docx
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
 
warnings.filterwarnings("ignore")
 
# ── Model (loaded once) ───────────────────────────────────────────────────────
print("[INFO] Loading Sentence-BERT model (first run downloads ~90MB)...")
MODEL = SentenceTransformer("all-MiniLM-L6-v2")
print("[INFO] Model ready.\n")
 
 
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  1. TEXT EXTRACTION                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝
 
def extract_pdf(path: str) -> str:
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text.strip()
 
 
def extract_docx(path: str) -> str:
    doc = docx.Document(path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
 
 
def extract_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().strip()
 
 
def extract_text(path: str) -> str:
    ext = path.lower().split(".")[-1]
    if ext == "pdf":
        return extract_pdf(path)
    elif ext in ("docx", "doc"):
        return extract_docx(path)
    else:
        return extract_txt(path)
 
 
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  2. SKILL KEYWORDS  (FIX: all lowercase, no leading spaces)             ║
# ╚══════════════════════════════════════════════════════════════════════════╝
 
# BUG FIXED: Original list had " opencvs", " spacy", "NLP" (uppercase) and
# " nltk" with leading spaces — none of these matched lowercased resume text.
# All entries are now stripped and lowercased so matching works correctly.
 
SKILL_KEYWORDS = [
    # General programming
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
    "kotlin", "swift",
    # Databases
    "sql", "nosql", "mongodb", "postgresql", "mysql", "redis", "elasticsearch",
    # Web / backend
    "react", "angular", "vue", "node", "django", "flask", "fastapi", "spring",
    # Cloud / DevOps
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ci/cd", "devops",
    # AI / ML core  ← was broken: "NLP" uppercase, " OpenCV" / " SpaCy" leading space
    "machine learning", "deep learning", "nlp", "computer vision",
    "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy",
    "data science", "analytics",
    # NLP / CV tooling  ← were broken: " opencvs", " spacy", " nltk", " openCV"
    "opencv", "tesseract", "transformers", "hugging face", "spacy", "nltk",
    # Specialised AI topics
    "sbert", "cosine similarity", "ocr", "resume parsing", "text classification",
    "lstm", "bert", "gpt",
    # Soft / process
    "agile", "scrum", "git", "linux", "rest api", "microservices", "graphql",
    "excel", "tableau", "power bi", "communication", "leadership", "teamwork",
    "project management", "problem solving", "research", "testing", "qa",
]
 
 
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  3. DEEP LEARNING SCORING                                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝
 
def chunk_text(text: str, chunk_size: int = 200) -> list[str]:
    """Split text into overlapping word chunks for better embedding coverage."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size // 2):
        chunk = " ".join(words[i: i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks or [text]
 
 
def semantic_score(jd_text: str, resume_text: str) -> float:
    """
    Deep Learning Step:
    Encode JD and resume into dense vector embeddings using Sentence-BERT,
    then compute cosine similarity across all chunk pairs and take the max
    average (best-alignment strategy).
    """
    jd_chunks     = chunk_text(jd_text, 200)
    resume_chunks = chunk_text(resume_text, 200)
 
    jd_embeddings     = MODEL.encode(jd_chunks,     convert_to_numpy=True, show_progress_bar=False)
    resume_embeddings = MODEL.encode(resume_chunks, convert_to_numpy=True, show_progress_bar=False)
 
    sim_matrix    = cosine_similarity(jd_embeddings, resume_embeddings)
    best_per_jd   = sim_matrix.max(axis=1)
    score         = float(np.mean(best_per_jd))
 
    normalized = max(0.0, min(1.0, (score - 0.2) / 0.75))
    return round(normalized * 100, 1)
 
 
def skills_score(jd_text: str, resume_text: str) -> tuple[float, list, list]:
    jd_lower     = jd_text.lower()
    resume_lower = resume_text.lower()

    required = [s for s in SKILL_KEYWORDS if s in jd_lower]
    if not required:
        return 50.0, [], []

    matched = []
    missing = []

    for s in required:
        found = False

        # ✅ exact match
        if s in resume_lower:
            found = True

        # ✅ handle space / hyphen differences
        elif s.replace(" ", "") in resume_lower.replace(" ", ""):
            found = True

        # ✅ simple partial word match
        else:
            for word in resume_lower.split():
                if s[:4] in word:
                    found = True
                    break

        if found:
            matched.append(s)
        else:
            missing.append(s)

    base_score = len(matched) / len(required) * 100

    # keep your semantic logic same
    skill_context_jd     = " ".join(required)
    skill_context_resume = resume_lower
    sem   = semantic_score(skill_context_jd, skill_context_resume)

    final = round(base_score * 0.6 + sem * 0.4, 1)
    return final, matched[:8], missing[:6]
 
 
def experience_score(jd_text: str, resume_text: str) -> float:
    """
    Estimate experience fit using SBERT on experience sentences + year heuristic.
    """
    exp_sentences_jd = [s for s in jd_text.split(".") if
                        any(w in s.lower() for w in
                            ["year", "experience", "worked", "background",
                             "senior", "junior", "mid"])]
    exp_sentences_rs = [s for s in resume_text.split(".") if
                        any(w in s.lower() for w in
                            ["year", "experience", "worked", "background",
                             "senior", "junior", "mid"])]
 
    if exp_sentences_jd and exp_sentences_rs:
        sem = semantic_score(" ".join(exp_sentences_jd), " ".join(exp_sentences_rs))
    else:
        sem = semantic_score(jd_text[:500], resume_text[:500])
 
    years_found = re.findall(r"(\d+)\s*\+?\s*year", resume_text.lower())
    year_boost  = min(10, sum(int(y) for y in years_found[:3]))
 
    return round(min(100, sem + year_boost), 1)
 
 
def education_score(jd_text: str, resume_text: str) -> float:
    """Semantic similarity focused on education-related content."""
    edu_words = ["degree", "bachelor", "master", "phd", "b.sc", "m.sc",
                 "b.e", "m.e", "university", "college", "diploma",
                 "certification", "graduate"]
 
    edu_jd = " ".join(s for s in jd_text.split(".")
                       if any(w in s.lower() for w in edu_words))
    edu_rs = " ".join(s for s in resume_text.split(".")
                       if any(w in s.lower() for w in edu_words))
 
    if not edu_jd or not edu_rs:
        return 60.0  # neutral when no education info is present
 
    return round(semantic_score(edu_jd, edu_rs), 1)
 
 
def extract_candidate_name(resume_text: str) -> str:
    """Heuristically extract candidate name from the top of the resume."""
    lines = [l.strip() for l in resume_text.split("\n") if l.strip()]
    for line in lines[:6]:
        words = line.split()
        if 2 <= len(words) <= 4 and all(w[0].isupper() for w in words if w.isalpha()):
            if not any(c.isdigit() for c in line):
                return line
    return "Candidate"
 
 
def verdict(score: float) -> str:
    if score >= 75: return "Strong match"
    if score >= 55: return "Good match"
    if score >= 35: return "Weak match"
    return "Poor match"
 
 
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  4. MAIN PIPELINE                                                        ║
# ║                                                                          ║
# ║  FIX: overall score is now a weighted blend of the three sub-scores.    ║
# ║  Before: overall = semantic_score(jd, resume)  ← pure SBERT cosine,    ║
# ║           completely ignoring skills/exp/edu sub-scores shown in UI.    ║
# ║  After:  overall = 0.45*skills + 0.35*experience + 0.20*education       ║
# ║           ranking now reflects actual resume content.                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝
 
WEIGHTS = {"skills": 0.45, "experience": 0.35, "education": 0.20}
 
 
def analyze_resume(jd_text: str, resume_path: str) -> dict:
    """Full analysis pipeline for one resume."""
    try:
        resume_text = extract_text(resume_path)
        if not resume_text:
            raise ValueError("Could not extract text from file.")
 
        sk, matched, missing = skills_score(jd_text, resume_text)
        exp = experience_score(jd_text, resume_text)
        edu = education_score(jd_text, resume_text)
 
        # FIXED: weighted blend instead of raw semantic cosine
        overall = round(
            WEIGHTS["skills"]     * sk  +
            WEIGHTS["experience"] * exp +
            WEIGHTS["education"]  * edu,
            1
        )
 
        name = extract_candidate_name(resume_text)
 
        return {
            "filename":   os.path.basename(resume_path),
            "name":       name,
            "overall":    overall,
            "skills":     sk,
            "experience": exp,
            "education":  edu,
            "verdict":    verdict(overall),
            "matched":    matched,
            "missing":    missing,
            "error":      None,
        }
    except Exception as e:
        return {
            "filename":   os.path.basename(resume_path),
            "name":       "Error",
            "overall":    0,
            "skills":     0,
            "experience": 0,
            "education":  0,
            "verdict":    "Error",
            "matched":    [],
            "missing":    [],
            "error":      str(e),
        }
 
 
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  5. GUI (Tkinter — runs natively in VS Code terminal)                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝
 
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI Resume Filter  —  Deep Learning")
        self.geometry("1000x750")
        self.configure(bg="#F8F8F8")
        self.resizable(True, True)
        self.resume_paths: list[str] = []
        self._build_ui()
 
    # ── UI construction ───────────────────────────────────────────────────
    def _build_ui(self):
        FONT   = ("Segoe UI", 10)
        FONT_B = ("Segoe UI", 10, "bold")
        BG     = "#F8F8F8"
        CARD   = "#FFFFFF"
        ACCENT = "#185FA5"
 
        tk.Label(self, text="AI Resume Filter", font=("Segoe UI", 16, "bold"),
                 bg=BG, fg="#111111").pack(pady=(16, 2))
        tk.Label(self,
                 text="Deep Learning  ·  Sentence-BERT  ·  Weighted Scoring",
                 font=("Segoe UI", 9), bg=BG, fg="#888888").pack(pady=(0, 12))
 
        pane = tk.Frame(self, bg=BG)
        pane.pack(fill="both", expand=True, padx=16, pady=0)
        pane.columnconfigure(0, weight=1)
        pane.columnconfigure(1, weight=1)
 
        # ── Left: Job Description ─────────────────────────────────────────
        left = tk.LabelFrame(pane, text=" Job Description ", font=FONT_B,
                              bg=CARD, relief="solid", bd=1)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=4)
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)
 
        self.jd_box = scrolledtext.ScrolledText(
            left, wrap="word", height=14, font=FONT, bg="#FAFAFA", relief="flat")
        self.jd_box.insert(
            "1.0",
            "Paste the job description here...\n\n"
            "Include required skills, experience level, responsibilities and qualifications.")
        self.jd_box.bind("<FocusIn>", self._clear_placeholder)
        self.jd_box.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
 
        # ── Right: Resume Upload ──────────────────────────────────────────
        right = tk.LabelFrame(pane, text=" Upload Resumes ", font=FONT_B,
                               bg=CARD, relief="solid", bd=1)
        right.grid(row=0, column=1, sticky="nsew", padx=(6, 0), pady=4)
        right.columnconfigure(0, weight=1)
 
        tk.Button(right, text="+ Browse Files  (PDF / DOCX / TXT)",
                  font=FONT_B, bg=ACCENT, fg="white", relief="flat",
                  activebackground="#0C447C", activeforeground="white",
                  padx=12, pady=6, cursor="hand2",
                  command=self._browse).grid(
                      row=0, column=0, sticky="ew", padx=8, pady=(10, 6))
 
        tk.Button(right, text="Clear list", font=FONT, bg="#EEEEEE",
                  relief="flat", cursor="hand2",
                  command=self._clear_files).grid(
                      row=1, column=0, sticky="ew", padx=8, pady=(0, 6))
 
        self.file_listbox = tk.Listbox(
            right, font=("Segoe UI", 9), selectmode="extended",
            bg="#FAFAFA", relief="flat", height=10, selectbackground="#B5D4F4")
        self.file_listbox.grid(row=2, column=0, sticky="nsew", padx=8, pady=(0, 8))
        right.rowconfigure(2, weight=1)
 
        self.file_count_lbl = tk.Label(
            right, text="0 file(s) loaded", font=("Segoe UI", 9), bg=CARD, fg="#888888")
        self.file_count_lbl.grid(row=3, column=0, pady=(0, 8))
 
        # ── Weights display ───────────────────────────────────────────────
        wf = tk.Frame(self, bg=BG)
        wf.pack()
        tk.Label(wf, text="Scoring weights:  ", font=("Segoe UI", 9),
                 bg=BG, fg="#888888").pack(side="left")
        for label, w in [("Skills 45%", WEIGHTS["skills"]),
                         ("Experience 35%", WEIGHTS["experience"]),
                         ("Education 20%", WEIGHTS["education"])]:
            tk.Label(wf, text=label, font=("Segoe UI", 9, "bold"),
                     bg=BG, fg="#185FA5").pack(side="left", padx=6)
 
        # ── Run button ────────────────────────────────────────────────────
        self.run_btn = tk.Button(
            self, text="🔍  Screen All Resumes",
            font=("Segoe UI", 11, "bold"),
            bg=ACCENT, fg="white", relief="flat",
            activebackground="#0C447C", activeforeground="white",
            padx=20, pady=8, cursor="hand2", command=self._run_thread)
        self.run_btn.pack(pady=10)
 
        self.prog_var = tk.DoubleVar()
        self.prog_bar = ttk.Progressbar(
            self, variable=self.prog_var, maximum=100, length=500)
        self.prog_bar.pack(pady=(0, 4))
        self.prog_lbl = tk.Label(
            self, text="", font=("Segoe UI", 9), bg=BG, fg="#555555")
        self.prog_lbl.pack()
 
        res_frame = tk.LabelFrame(
            self, text=" Results ", font=FONT_B, bg=CARD, relief="solid", bd=1)
        res_frame.pack(fill="both", expand=True, padx=16, pady=(6, 12))
        res_frame.rowconfigure(0, weight=1)
        res_frame.columnconfigure(0, weight=1)
 
        self.result_box = scrolledtext.ScrolledText(
            res_frame, wrap="word", font=("Courier New", 9),
            bg="#FAFAFA", relief="flat", state="disabled")
        self.result_box.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        self._tag_colors()
 
    def _tag_colors(self):
        self.result_box.tag_config(
            "header",  foreground="#185FA5", font=("Courier New", 10, "bold"))
        self.result_box.tag_config(
            "rank",    foreground="#854F0B", font=("Courier New", 10, "bold"))
        self.result_box.tag_config("high",    foreground="#3B6D11")
        self.result_box.tag_config("mid",     foreground="#185FA5")
        self.result_box.tag_config("low",     foreground="#854F0B")
        self.result_box.tag_config("poor",    foreground="#A32D2D")
        self.result_box.tag_config("hit",     foreground="#3B6D11")
        self.result_box.tag_config("miss",    foreground="#A32D2D")
        self.result_box.tag_config("err",     foreground="#A32D2D")
        self.result_box.tag_config("divider", foreground="#AAAAAA")
 
    # ── Helpers ───────────────────────────────────────────────────────────
    def _clear_placeholder(self, event):
        content = self.jd_box.get("1.0", "end-1c")
        if content.startswith("Paste the job description"):
            self.jd_box.delete("1.0", tk.END)
 
    def _browse(self):
        paths = filedialog.askopenfilenames(
            title="Select Resume Files",
            filetypes=[("Resume files", "*.pdf *.docx *.doc *.txt"),
                       ("All files", "*.*")])
        for p in paths:
            if p not in self.resume_paths:
                self.resume_paths.append(p)
                self.file_listbox.insert(tk.END, f"  {os.path.basename(p)}")
        self.file_count_lbl.config(
            text=f"{len(self.resume_paths)} file(s) loaded")
 
    def _clear_files(self):
        self.resume_paths.clear()
        self.file_listbox.delete(0, tk.END)
        self.file_count_lbl.config(text="0 file(s) loaded")
 
    def _set_result(self, text: str, tag: str = ""):
        self.result_box.config(state="normal")
        if tag:
            self.result_box.insert(tk.END, text, tag)
        else:
            self.result_box.insert(tk.END, text)
        self.result_box.see(tk.END)
        self.result_box.config(state="disabled")
 
    def _clear_result(self):
        self.result_box.config(state="normal")
        self.result_box.delete("1.0", tk.END)
        self.result_box.config(state="disabled")
 
    # ── Run (threaded) ────────────────────────────────────────────────────
    def _run_thread(self):
        threading.Thread(target=self._run, daemon=True).start()
 
    def _run(self):
        jd = self.jd_box.get("1.0", "end-1c").strip()
        if not jd or jd.startswith("Paste the job description"):
            self._show_error("Please paste a job description first.")
            return
        if not self.resume_paths:
            self._show_error("Please upload at least one resume file.")
            return
 
        self.run_btn.config(state="disabled", text="Analyzing…")
        self._clear_result()
        self.prog_var.set(0)
 
        results = []
        total   = len(self.resume_paths)
 
        for i, path in enumerate(self.resume_paths):
            fname = os.path.basename(path)
            self.prog_lbl.config(text=f"Processing {fname}  ({i+1}/{total})")
            result = analyze_resume(jd, path)
            results.append(result)
            self.prog_var.set((i + 1) / total * 100)
 
        self.prog_lbl.config(text="Done!")
        self._render_results(
            sorted(results, key=lambda r: r["overall"], reverse=True))
        self.run_btn.config(state="normal", text="🔍  Screen All Resumes")
 
    def _render_results(self, results: list[dict]):
        medals  = ["🥇", "🥈", "🥉"]
        SUCCESS = [r for r in results if not r["error"]]
        FAILED  = [r for r in results if r["error"]]
 
        self._set_result("━" * 72 + "\n", "divider")
        self._set_result(
            f"  RESULTS  —  {len(SUCCESS)} resume(s) screened\n", "header")
        self._set_result(
            f"  Weights: Skills {int(WEIGHTS['skills']*100)}%  |  "
            f"Experience {int(WEIGHTS['experience']*100)}%  |  "
            f"Education {int(WEIGHTS['education']*100)}%\n", "header")
        self._set_result("━" * 72 + "\n", "divider")
 
        for i, r in enumerate(SUCCESS):
            medal = medals[i] if i < 3 else f"#{i+1}"
            score = r["overall"]
            tag   = ("high" if score >= 75 else
                     "mid"  if score >= 55 else
                     "low"  if score >= 35 else "poor")
 
            self._set_result(f"\n  {medal}  Rank {i+1}  —  ", "rank")
            self._set_result(f"{r['name']}  ", "header")
            self._set_result(f"[ {score}% — {r['verdict']} ]\n", tag)
            self._set_result(f"     File: {r['filename']}\n")
 
            for label, val in [("Skills    ", r["skills"]),
                                ("Experience", r["experience"]),
                                ("Education ", r["education"])]:
                bar_len = int(val / 5)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                t   = ("high" if val >= 75 else
                       "mid"  if val >= 55 else
                       "low"  if val >= 35 else "poor")
                self._set_result(f"     {label}  {bar}  {val:.0f}%\n", t)
 
            if r["matched"]:
                self._set_result("     ✔  Matched  : ", "hit")
                self._set_result(", ".join(r["matched"]) + "\n")
            if r["missing"]:
                self._set_result("     ✘  Missing  : ", "miss")
                self._set_result(", ".join(r["missing"]) + "\n")
 
            self._set_result("─" * 72 + "\n", "divider")
 
        if FAILED:
            self._set_result("\n  Errors:\n", "err")
            for r in FAILED:
                self._set_result(
                    f"  ✘ {r['filename']}  —  {r['error']}\n", "err")
 
    def _show_error(self, msg: str):
        self._clear_result()
        self._set_result(f"\n  ✘  {msg}\n", "err")
 
 
# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = App()
    app.mainloop()