"""Tests for the shared text-processing foundation (paperscope/text/).

Covers the layers the existing tests/test_text_parsing.py does not:

- ``latex.clean_latex`` / ``latex.extract_cite_keys`` / ``latex.clean_plaintext``
- ``chunking.chunk_text`` (overlap, word-count bounds, tail handling)
- the parsing helpers built on top (citation contexts, claims, paragraph
  line numbers) on small synthetic inputs.

Two defects originally pinned here as strict xfails (unstripped ``%``
comments; plain-newline de-hyphenation) have since been fixed in
``text/latex.py`` — their tests now assert the fixed behaviour directly.
"""

from __future__ import annotations

import pytest

from paperscope.text.chunking import chunk_text
from paperscope.text.latex import clean_latex, clean_plaintext, extract_cite_keys
from paperscope.text.parsing import (
    extract_citation_contexts,
    extract_claims,
    extract_paragraphs,
    extract_sections,
    split_sentences,
)


# ---------------------------------------------------------------------------
# clean_latex
# ---------------------------------------------------------------------------


class TestCleanLatex:
    def test_citation_and_ref_commands_removed_entirely(self):
        out = clean_latex(
            r"Result shown in \cite{smith2020} and \ref{fig:1}, see \label{sec:x}."
        )
        assert "smith2020" not in out
        assert "fig:1" not in out
        assert "\\" not in out

    def test_inline_math_becomes_placeholder(self):
        out = clean_latex(r"Energy $E = mc^2$ is conserved.")
        assert "MATH" in out
        assert "$" not in out
        assert "mc" not in out

    def test_environment_markers_removed_content_kept(self):
        out = clean_latex(r"\begin{itemize}\item first point kept\end{itemize}")
        assert out == "first point kept"

    def test_nested_commands_unwrapped(self):
        assert clean_latex(r"\textbf{Bold \emph{nested}} rest") == "Bold nested rest"

    def test_href_keeps_link_text_drops_url(self):
        out = clean_latex(r"see \href{http://x.com}{the site} ok")
        assert out == "see the site ok"

    def test_bare_commands_removed(self):
        out = clean_latex(r"one \noindent two \newline three")
        assert out == "one two three"

    def test_urls_and_bare_dois_stripped(self):
        out = clean_latex("visit https://example.com/page now, doi 10.1234/j.x.5 gone")
        assert out == "visit now, doi gone"

    def test_dashes_normalised(self):
        out = clean_latex("range 1--2 and dash --- here")
        assert out == "range 1–2 and dash — here"

    def test_special_characters_and_whitespace_normalised(self):
        out = clean_latex("a_b^c & d   \n\n e")
        assert out == "a b c d e"

    def test_tilde_becomes_space(self):
        assert clean_latex("Figure~3") == "Figure 3"

    def test_empty_input(self):
        assert clean_latex("") == ""

    def test_percent_comments_stripped(self):
        out = clean_latex("Keep this. % secret draft note\nAnd this.")
        assert "secret" not in out
        assert "Keep this." in out and "And this." in out


# ---------------------------------------------------------------------------
# extract_cite_keys
# ---------------------------------------------------------------------------


class TestExtractCiteKeys:
    def test_variants_multikeys_order_and_duplicates(self):
        keys = extract_cite_keys(r"\citep{a, b}\citet{c}\cite{a}\cite{*}")
        assert keys == ["a", "b", "c", "a"]

    def test_citeauthor_variant(self):
        assert extract_cite_keys(r"\citeauthor{knuth1984}") == ["knuth1984"]

    def test_no_citations_returns_empty(self):
        assert extract_cite_keys("plain text without citations") == []

    def test_whitespace_around_keys_stripped(self):
        assert extract_cite_keys(r"\cite{ x ,y }") == ["x", "y"]


# ---------------------------------------------------------------------------
# clean_plaintext
# ---------------------------------------------------------------------------


class TestCleanPlaintext:
    def test_form_feed_removed_and_whitespace_collapsed(self):
        assert clean_plaintext("a\x0cb\n\n  c\td") == "a b c d"

    def test_dehyphenation_with_crlf_line_break(self):
        assert clean_plaintext("exam- \nple and hy-\r\nphen") == "example and hyphen"

    def test_dehyphenation_with_plain_newline(self):
        assert clean_plaintext("exam-\nple") == "example"


# ---------------------------------------------------------------------------
# chunk_text
# ---------------------------------------------------------------------------


class TestChunkText:
    def test_empty_and_whitespace_input(self):
        assert chunk_text("") == []
        assert chunk_text("   \n\t ") == []

    def test_short_text_returned_as_single_chunk(self):
        assert chunk_text("hello world", target_words=10) == ["hello world"]

    def test_exact_target_length_is_single_chunk(self):
        words = [f"w{i}" for i in range(10)]
        assert chunk_text(" ".join(words), target_words=10) == [" ".join(words)]

    def test_chunks_respect_target_word_bound(self):
        words = [f"w{i:02d}" for i in range(25)]
        chunks = chunk_text(
            " ".join(words), target_words=10, overlap_words=3, min_chunk_words=2
        )
        assert all(len(c.split()) <= 10 for c in chunks)

    def test_consecutive_chunks_share_overlap_words(self):
        words = [f"w{i:02d}" for i in range(25)]
        chunks = chunk_text(
            " ".join(words), target_words=10, overlap_words=3, min_chunk_words=2
        )
        assert len(chunks) == 4
        for prev, nxt in zip(chunks, chunks[1:]):
            assert prev.split()[-3:] == nxt.split()[:3]

    def test_no_dropped_tail_when_above_minimum(self):
        words = [f"w{i:02d}" for i in range(25)]
        chunks = chunk_text(
            " ".join(words), target_words=10, overlap_words=3, min_chunk_words=2
        )
        covered = {w for c in chunks for w in c.split()}
        assert covered == set(words)
        assert chunks[-1].split()[-1] == words[-1]

    def test_tail_below_min_chunk_words_is_dropped_by_contract(self):
        # Documented: "Discard trailing chunks smaller than min_chunk_words".
        words = [f"x{i}" for i in range(21)]
        chunks = chunk_text(
            " ".join(words), target_words=10, overlap_words=0, min_chunk_words=5
        )
        assert chunks == [
            " ".join(words[0:10]),
            " ".join(words[10:20]),
        ]

    def test_overlap_ge_target_still_terminates(self):
        # step = max(target - overlap, 1) floors at 1: no infinite loop.
        words = [f"y{i}" for i in range(25)]
        chunks = chunk_text(
            " ".join(words), target_words=5, overlap_words=7, min_chunk_words=1
        )
        assert chunks
        assert all(len(c.split()) <= 5 for c in chunks)
        assert chunks[-1].split()[-1] == words[-1]


# ---------------------------------------------------------------------------
# section parsing (complementary to tests/test_text_parsing.py)
# ---------------------------------------------------------------------------


SECTIONED = r"""
\documentclass{article}
\begin{document}
\section{Introduction}
Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu.

\subsection{Background}
One two three four five six seven eight nine ten eleven twelve.

\section*{Starred Section}
Uno dos tres cuatro cinco seis siete ocho nueve diez once doce.
\end{document}
"""


class TestExtractSections:
    def test_section_subsection_and_starred_all_captured_in_order(self):
        sections = extract_sections(SECTIONED, min_words=5)
        assert [s["title"] for s in sections] == [
            "Introduction",
            "Background",
            "Starred Section",
        ]

    def test_section_text_is_cleaned_plain_text(self):
        sections = extract_sections(SECTIONED, min_words=5)
        for s in sections:
            assert "\\" not in s["text"]
            assert len(s["text"].split()) >= 5


# ---------------------------------------------------------------------------
# split_sentences
# ---------------------------------------------------------------------------


class TestSplitSentences:
    def test_splits_before_capital_or_backslash(self):
        parts = split_sentences(r"First sentence ends. \textbf{Bold} starts next.")
        assert parts == ["First sentence ends.", r"\textbf{Bold} starts next."]

    def test_lowercase_continuation_not_split(self):
        parts = split_sentences("Version 2.0 was released. next words stay attached")
        assert len(parts) == 1

    def test_question_and_exclamation_split(self):
        parts = split_sentences("Really? Yes! Definitely.")
        assert parts == ["Really?", "Yes!", "Definitely."]


# ---------------------------------------------------------------------------
# extract_paragraphs (line numbers + filtering)
# ---------------------------------------------------------------------------


PARA_DOC = "\n".join(
    [
        r"\documentclass{article}",  # line 1
        r"\usepackage{foo}",  # line 2
        "",  # line 3
        r"\begin{document}",  # line 4
        "",  # line 5
        "First paragraph spans two source lines and easily",  # line 6
        "clears the minimum word threshold for inclusion.",  # line 7
        "",  # line 8
        "Tiny stub.",  # line 9
        "",  # line 10
        "Final paragraph also has comfortably more than the minimum words.",  # 11
    ]
)


class TestExtractParagraphs:
    def test_line_numbers_reference_original_tex_file(self):
        paras = extract_paragraphs(PARA_DOC, min_words=5)
        assert [p["line"] for p in paras] == [6, 11]

    def test_short_paragraphs_filtered_and_text_joined(self):
        paras = extract_paragraphs(PARA_DOC, min_words=5)
        assert all("Tiny stub" not in p["text"] for p in paras)
        assert paras[0]["text"].startswith("First paragraph spans two source lines")
        assert "threshold for inclusion." in paras[0]["text"]

    def test_trailing_paragraph_without_blank_line_is_flushed(self):
        paras = extract_paragraphs(PARA_DOC, min_words=5)
        assert paras[-1]["text"].startswith("Final paragraph")


# ---------------------------------------------------------------------------
# extract_citation_contexts
# ---------------------------------------------------------------------------


CITED_DOC = "\n".join(
    [
        r"\begin{document}",  # line 1
        "",  # line 2
        r"Alpha beta gamma delta results follow \cite{smith2020, jones2019}.",  # 3
        "Second sentence here has no citations at all.",  # line 4
        "",  # line 5
        "Long context sentence with many descriptive words here.",  # line 6
        r"See \cite{brown2021}.",  # line 7
        "",  # line 8
        r"\begin{thebibliography}{9}",  # line 9
        r"\bibitem{smith2020} Smith. \cite{never_counted}",  # line 10
        r"\end{thebibliography}",  # line 11
    ]
)


class TestExtractCitationContexts:
    def test_ids_are_sequential_and_keys_captured(self):
        ctxs = extract_citation_contexts(CITED_DOC)
        assert [c["id"] for c in ctxs] == ["CTX01", "CTX02"]
        assert ctxs[0]["cited_keys"] == ["smith2020", "jones2019"]
        assert ctxs[1]["cited_keys"] == ["brown2021"]

    def test_line_numbers_point_at_paragraph_start(self):
        ctxs = extract_citation_contexts(CITED_DOC)
        assert ctxs[0]["line"] == 3
        assert ctxs[1]["line"] == 6

    def test_text_is_cleaned_and_bibliography_excluded(self):
        ctxs = extract_citation_contexts(CITED_DOC)
        for c in ctxs:
            assert "\\" not in c["text"]
        assert all("never_counted" not in c["cited_keys"] for c in ctxs)

    def test_short_citing_sentence_falls_back_to_paragraph_context(self):
        ctxs = extract_citation_contexts(CITED_DOC)
        # "See ." is < 4 cleaned words, so the whole paragraph is used.
        assert ctxs[1]["text"].startswith("Long context sentence")


# ---------------------------------------------------------------------------
# extract_claims
# ---------------------------------------------------------------------------


CLAIMS_DOC = "\n".join(
    [
        "Intro line without claims.",  # line 1
        r"\textbf{Key claim with several words} follows.",  # line 2
        r"\textbf{tiny} is skipped.",  # line 3
        r"\textbf{Another substantive bolded claim here} too.",  # line 4
    ]
)


class TestExtractClaims:
    def test_bold_claims_of_three_plus_words_captured(self):
        claims = extract_claims(CLAIMS_DOC)
        assert [c["text"] for c in claims] == [
            "Key claim with several words",
            "Another substantive bolded claim here",
        ]
        assert all(c["type"] == "bold_claim" for c in claims)

    def test_ids_and_line_numbers(self):
        claims = extract_claims(CLAIMS_DOC)
        assert [c["id"] for c in claims] == ["CLM01", "CLM02"]
        assert [c["line"] for c in claims] == [2, 4]

    def test_short_bold_text_not_a_claim(self):
        claims = extract_claims(CLAIMS_DOC)
        assert all("tiny" not in c["text"] for c in claims)


# ---------------------------------------------------------------------------
# adversarial-review regressions: comment order (B2) + dehyphenation (B3)
# ---------------------------------------------------------------------------


class TestCleanLatexCommentOrder:
    """Comment stripping must run AFTER \\url/\\href/cite removal: a raw %
    inside \\url{...} is part of the URL, not a comment, and must not
    truncate the rest of the line."""

    def test_percent_inside_url_does_not_truncate_line(self):
        out = clean_latex(r"See \url{http://x.org/a%b} for details. % note")
        assert "for details." in out
        assert "note" not in out  # the real trailing comment still goes

    def test_percent_inside_href_url_does_not_truncate_line(self):
        out = clean_latex(r"see \href{http://x.com/a%20b}{the site} stays put")
        assert "stays put" in out and "the site" in out

    def test_plain_comments_still_stripped(self):
        out = clean_latex("Keep this. % secret draft note\nAnd this.")
        assert "secret" not in out and "Keep this." in out and "And this." in out


class TestCleanPlaintextDehyphenation:
    """Dehyphenation is for PDF soft-wrapped words (letter before the hyphen,
    lowercase letter after the break) — it must not merge numeric ranges
    split across lines ('12-\\n15' must never become '1215')."""

    def test_numeric_range_not_merged(self):
        out = clean_plaintext("pages 12-\n15 of the report")
        assert "1215" not in out
        assert "12-" in out and "15" in out

    def test_word_break_still_rejoined(self):
        assert clean_plaintext("hyphen-\nated") == "hyphenated"

    def test_crlf_and_space_variants_still_work(self):
        assert clean_plaintext("exam- \nple and hy-\r\nphen") == "example and hyphen"

    def test_mixed_line(self):
        out = clean_plaintext("see pp. 12-\n15 for the hyphen-\nated case")
        assert "1215" not in out and "hyphenated" in out
