import streamlit as st
from openai import OpenAI
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Headless backend for Streamlit Cloud
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
import random
import json
import re
from fpdf import FPDF
from io import BytesIO

# --- Initialize OpenAI client ---
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# --- Streamlit setup ---
st.set_page_config(page_title="Kids Crossword Generator", page_icon="🧩", layout="centered")
st.title("🧩 Kids Crossword Generator – Visual Version")
st.caption("Create fun, printable, multilingual crossword puzzles for students!")

# --- Inputs ---
language = st.selectbox("🌍 Choose language", ["English", "Spanish", "French"])
topic = st.text_input("🎨 Enter a topic", "Animals")
age = st.slider("👶 Age group", 5, 10, 7)
grid_size = st.selectbox("📏 Grid size", [8, 10, 12, 14, 16, 18, 20])

# --- Function to draw crossword as image ---
def draw_crossword(grid, show_letters=False):
    n_rows, n_cols = grid.shape
    fig, ax = plt.subplots(figsize=(n_cols, n_rows), dpi=100)

    # Draw grid lines
    for x in range(n_cols + 1):
        ax.plot([x, x], [0, n_rows], color="black", linewidth=2)
    for y in range(n_rows + 1):
        ax.plot([0, n_cols], [y, y], color="black", linewidth=2)

    # Fill letters
    if show_letters:
        for i in range(n_rows):
            for j in range(n_cols):
                if grid[i, j] != "_":
                    ax.text(j + 0.5, n_rows - i - 0.5, grid[i, j],
                            ha="center", va="center", fontsize=20)

    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.axis("off")
    plt.tight_layout()

    # Save figure to in-memory PNG and open as PIL Image
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)
    return img

# --- Generate crossword ---
if st.button("✨ Generate Crossword"):
    with st.spinner("Asking AI for kid-friendly words and clues..."):
        prompt = f"""
        Generate {grid_size//2 + 2} simple words and clues for a crossword puzzle
        for children aged {age}. 
        Language: {language}.
        Theme: {topic}.
        Each word should be 3–8 letters long.
        Return ONLY valid JSON, with this exact structure:
        {{"words": ["WORD1", "WORD2", ...], "clues": ["CLUE1", "CLUE2", ...]}}
        Do not include any text outside the JSON.
        """
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        raw = response.choices[0].message.content.strip()

        # Extract JSON from response
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            raw_json = match.group(0)
        else:
            st.error("⚠️ No JSON found in AI response.")
            st.text(raw)
            st.stop()
        try:
            data = json.loads(raw_json)
            words = [w.upper() for w in data["words"]]
            clues = data["clues"]
        except Exception as e:
            st.error("⚠️ Couldn't parse AI response as JSON.")
            st.text(raw)
            st.stop()

    # --- Crossword logic ---
    def find_crossing_positions(word, grid):
        positions = []
        n_rows, n_cols = grid.shape
    
        for r in range(n_rows):
            for c in range(n_cols):
                letter = grid[r, c]
    
                if letter == ".":
                    continue
    
                for i, ch in enumerate(word):
                    if ch == letter:
                        # horizontal word crossing vertical
                        positions.append((r, c, "H", i))
                        # vertical word crossing horizontal
                        positions.append((r, c, "V", i))

    return positions

    def place_word_crossing(word, grid):
        positions = find_crossing_positions(word, grid)
        random.shuffle(positions)
    
        n_rows, n_cols = grid.shape
    
        for r, c, direction, offset in positions:
    
            if direction == "H":
                start_c = c - offset
                if start_c < 0 or start_c + len(word) > n_cols:
                    continue
    
                ok = True
                for i, ch in enumerate(word):
                    cell = grid[r, start_c + i]
                    if cell not in [".", ch]:
                        ok = False
                        break
    
                if ok:
                    for i, ch in enumerate(word):
                        grid[r, start_c + i] = ch
                    return True
    
            if direction == "V":
                start_r = r - offset
                if start_r < 0 or start_r + len(word) > n_rows:
                    continue
    
                ok = True
                for i, ch in enumerate(word):
                    cell = grid[start_r + i, c]
                    if cell not in [".", ch]:
                        ok = False
                        break
    
                if ok:
                    for i, ch in enumerate(word):
                        grid[start_r + i, c] = ch
                    return True

        return False
        
    grid = np.full((grid_size, grid_size), ".", dtype=str)

    def can_place(word, x, y, direction):
        if direction == "H" and y + len(word) <= grid_size:
            return all(grid[x, y + i] in [".", ch] for i, ch in enumerate(word))
        if direction == "V" and x + len(word) <= grid_size:
            return all(grid[x + i, y] in [".", ch] for i, ch in enumerate(word))
        return False

    def place_word(word, grid):
        """
        Try to place word crossing letters if possible.
        Fallback to random placement if no crossing position is available.
        """
        if place_word_crossing(word, grid):
            return True  # placed crossing existing letters
    
        # fallback: random placement
        n_rows, n_cols = grid.shape
        for _ in range(100):
            direction = random.choice(["H", "V"])
            x = random.randint(0, n_rows - 1)
            y = random.randint(0, n_cols - 1)
            if direction == "H" and y + len(word) <= n_cols:
                if all(grid[x, y + i] in [".", word[i]] for i in range(len(word))):
                    for i, ch in enumerate(word):
                        grid[x, y + i] = ch
                    return True
            elif direction == "V" and x + len(word) <= n_rows:
                if all(grid[x + i, y] in [".", word[i]] for i in range(len(word))):
                    for i, ch in enumerate(word):
                        grid[x + i, y] = ch
                    return True
        return False

    placed = []
    # place first word in center
    first = words[0]
    row = grid_size // 2
    col = (grid_size - len(first)) // 2
    
    for i, ch in enumerate(first):
        grid[row, col + i] = ch
    
    placed = [first]

    blank_grid = np.where(grid == ".", ".", "_")

    # --- Render visual grids ---
    puzzle_img = draw_crossword(blank_grid, show_letters=False)
    answer_img = draw_crossword(grid, show_letters=True)

    st.image(puzzle_img, caption="🧩 Crossword Puzzle", use_column_width=True)
    st.image(answer_img, caption="✅ Answer Key", use_column_width=True)

    st.subheader("🪄 Clues")
    for i, c in enumerate(clues, 1):
        st.write(f"{i}. {c}")

    # --- Generate PDF ---
    pdf_name = f"crossword_{topic}_{language}.pdf"
    pdf = FPDF()
    pdf.add_page()

    # Save images temporarily
    puzzle_img_path = "puzzle.png"
    answer_img_path = "answer.png"
    puzzle_img.save(puzzle_img_path)
    answer_img.save(answer_img_path)

    # Insert puzzle + clues
    pdf.image(puzzle_img_path, x=10, y=20, w=pdf.w - 20)
    pdf.set_font("DejaVu", "", 12)
    pdf.ln(10)
    for i, c in enumerate(clues, 1):
        pdf.multi_cell(0, 8, f"{i}. {c}")

    # Insert answer key page
    pdf.add_page()
    pdf.image(answer_img_path, x=10, y=20, w=pdf.w - 20)

    # Output PDF
    pdf.output(pdf_name)

    with open(pdf_name, "rb") as f:
        st.download_button("📄 Download Printable PDF", f, file_name=pdf_name, mime="application/pdf")

    st.success("✅ Crossword with answer key generated successfully!")






