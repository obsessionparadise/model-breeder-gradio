"""
Model Breeder — Material 3 theme
==================================
A hand-built Material Design 3 token system for the Gradio app: a tonal
color palette (primary/secondary/neutral, each as a full 50-950 ramp),
Material's own type family (Roboto for UI, Roboto Mono for technical
values — seeds, filenames, block-weight vectors), a rounded shape scale,
and Material elevation (soft layered shadows instead of hard borders).

Two things are exported:
  - MATERIAL_THEME: a gr.Theme, passed to gr.Blocks(theme=...)
  - MATERIAL_CSS:   supplemental CSS for things the theme API can't reach
                     (cards, app bar, chips, tab indicator, progress bars,
                     console-style log panels), written against elem_classes
                     this app assigns itself, plus Gradio's own stable
                     structural selectors (.tabs / .tab-nav / .tabitem).
"""
import gradio as gr

# ─────────────────────────────────────────────────────────────────────────
# Color tokens — hand-tuned tonal ramps (Material 3 shape: 50 lightest,
# 950 darkest, 500ish = the "key color" used for filled surfaces).
# ─────────────────────────────────────────────────────────────────────────

# Primary — "Synthesis Violet": the merge/breed action color.
PRIMARY = gr.themes.Color(
    name="synthesis_violet",
    c50="#F5F2FF", c100="#ECE6FF", c200="#D9CCFF", c300="#BFA6FF",
    c400="#A47EFF", c500="#7C5CFA", c600="#6743E8", c700="#5533C4",
    c800="#44279E", c900="#351D7A", c950="#221252",
)

# Secondary — "Catalyst Amber": generation / bake actions, warnings-adjacent
# warmth that reads as "energy applied to the model," not an alert color.
SECONDARY = gr.themes.Color(
    name="catalyst_amber",
    c50="#FFF8E8", c100="#FFEFC6", c200="#FFDD8A", c300="#FFC94D",
    c400="#FFB627", c500="#F59E0B", c600="#D3830A", c700="#A8630A",
    c800="#7C4A0C", c900="#5C380D", c950="#341F06",
)

# Neutral — violet-tinted gray for surfaces (Material 3 surfaces always
# carry a whisper of the primary hue rather than being true gray).
NEUTRAL = gr.themes.Color(
    name="tinted_surface",
    c50="#FCFAFF", c100="#F5F2FA", c200="#ECE7F3", c300="#DED7E8",
    c400="#C6BDD4", c500="#9C93AC", c600="#756C86", c700="#574F68",
    c800="#3B3550", c900="#241F35", c950="#14111F",
)

# A few raw hex tokens used directly in custom CSS (elevation shadows,
# state layers, semantic status colors) that don't map onto the Color
# ramps above.
TOKENS = {
    "surface": "#FCFAFF",
    "surface_container_low": "#F7F2FC",
    "surface_container": "#F1ECF7",
    "surface_container_high": "#EBE5F3",
    "surface_container_highest": "#E5DEEE",
    "on_surface": "#1C1B1F",
    "on_surface_variant": "#48454E",
    "outline": "#D8D2E3",
    "outline_variant": "#E7E1F0",
    "primary": "#6B4FE0",
    "on_primary": "#FFFFFF",
    "primary_container": "#E6DFFF",
    "on_primary_container": "#22105E",
    "secondary": "#B4780A",
    "on_secondary": "#FFFFFF",
    "secondary_container": "#FFEBC2",
    "on_secondary_container": "#402D00",
    "success": "#1E8E5A",
    "success_container": "#DCF5E7",
    "on_success_container": "#0B3D24",
    "error": "#BA1A1A",
    "error_container": "#FFDAD6",
    "on_error_container": "#410002",
}

# ─────────────────────────────────────────────────────────────────────────
# Typography — Roboto is Material's own body/UI face; Roboto Mono for
# technical data (seeds, filenames, block-weight vectors, logs).
# ─────────────────────────────────────────────────────────────────────────
DISPLAY_FONT = gr.themes.GoogleFont("Roboto")
MONO_FONT = gr.themes.GoogleFont("Roboto Mono")


def build_theme() -> gr.Theme:
    theme = gr.themes.Base(
        primary_hue=PRIMARY,
        secondary_hue=SECONDARY,
        neutral_hue=NEUTRAL,
        spacing_size=gr.themes.sizes.spacing_md,
        radius_size=gr.themes.sizes.radius_lg,
        font=(DISPLAY_FONT, "ui-sans-serif", "system-ui", "sans-serif"),
        font_mono=(MONO_FONT, "ui-monospace", "Consolas", "monospace"),
    ).set(
        # ── Page / body ──────────────────────────────────────────────
        body_background_fill=TOKENS["surface"],
        body_text_color=TOKENS["on_surface"],
        body_text_size="15px",
        body_text_weight="400",
        color_accent=TOKENS["primary"],
        link_text_color=TOKENS["primary"],
        link_text_color_hover=TOKENS["primary"],

        # ── Blocks / cards (Material "surface containers") ─────────────
        block_background_fill=TOKENS["surface_container_low"],
        block_border_width="0px",
        block_border_color=TOKENS["outline_variant"],
        block_radius="20px",
        block_padding="20px",
        block_shadow="0 1px 2px 0 rgba(34,18,82,0.06), 0 1px 3px 1px rgba(34,18,82,0.08)",
        block_label_background_fill="transparent",
        block_label_text_color=TOKENS["on_surface_variant"],
        block_label_text_size="12px",
        block_label_text_weight="600",
        block_label_radius="8px",
        block_label_border_width="0px",
        block_title_text_color=TOKENS["on_surface"],
        block_title_text_weight="600",
        block_info_text_color=TOKENS["on_surface_variant"],
        panel_background_fill=TOKENS["surface_container"],
        panel_border_width="0px",
        container_radius="20px",

        # ── Shape scale (rounded everywhere, Material "full" pills for
        #    small controls, large-radius for containers) ───────────────
        input_radius="14px",
        table_radius="16px",
        embed_radius="20px",
        checkbox_border_radius="8px",
        button_large_radius="999px",
        button_medium_radius="999px",
        button_small_radius="999px",

        # ── Inputs (Material "filled text field" feel: soft fill,
        #    no boxy border, color shift on focus) ───────────────────────
        input_background_fill=TOKENS["surface_container_high"],
        input_background_fill_hover=TOKENS["surface_container_highest"],
        input_background_fill_focus=TOKENS["surface"],
        input_border_color="transparent",
        input_border_color_hover=TOKENS["outline"],
        input_border_color_focus=TOKENS["primary"],
        input_border_width="2px",
        input_shadow="none",
        input_shadow_focus="0 0 0 1px " + TOKENS["primary"],
        input_padding="10px 14px",
        input_placeholder_color=TOKENS["on_surface_variant"],

        # ── Slider (used both as a real control and, non-interactively,
        #    as a Material linear-progress indicator) ────────────────────
        slider_color=TOKENS["primary"],

        # ── Buttons — Material 3 "filled" (primary) / "tonal" (secondary)
        #    button styles, fully rounded, state-layer hover ─────────────
        button_primary_background_fill=TOKENS["primary"],
        button_primary_background_fill_hover=PRIMARY.c600,
        button_primary_text_color=TOKENS["on_primary"],
        button_primary_border_color=TOKENS["primary"],
        button_primary_border_color_hover=PRIMARY.c600,
        button_primary_shadow="0 1px 2px 0 rgba(34,18,82,0.15)",
        button_primary_shadow_hover="0 2px 6px 0 rgba(34,18,82,0.25)",

        button_secondary_background_fill=TOKENS["primary_container"],
        button_secondary_background_fill_hover="#DACCFF",
        button_secondary_text_color=TOKENS["on_primary_container"],
        button_secondary_border_color="transparent",
        button_secondary_border_color_hover="transparent",
        button_secondary_shadow="none",
        button_secondary_shadow_hover="none",

        button_cancel_background_fill=TOKENS["error_container"],
        button_cancel_background_fill_hover="#FFC7C2",
        button_cancel_text_color=TOKENS["on_error_container"],
        button_cancel_border_color="transparent",

        # ── Checkboxes / radios ─────────────────────────────────────────
        checkbox_background_color=TOKENS["surface_container_high"],
        checkbox_background_color_selected=TOKENS["primary"],
        checkbox_border_color=TOKENS["outline"],
        checkbox_border_color_selected=TOKENS["primary"],
        checkbox_border_width="2px",
        checkbox_label_background_fill=TOKENS["surface_container_high"],
        checkbox_label_background_fill_selected=TOKENS["primary_container"],
        checkbox_label_text_color=TOKENS["on_surface"],
        checkbox_label_text_color_selected=TOKENS["on_primary_container"],
        checkbox_label_border_color="transparent",
        checkbox_label_border_color_hover=TOKENS["outline"],
        checkbox_label_shadow="none",

        # ── Misc ─────────────────────────────────────────────────────────
        shadow_drop="0 1px 2px 0 rgba(34,18,82,0.08)",
        shadow_drop_lg="0 2px 8px 0 rgba(34,18,82,0.12)",
        border_color_primary=TOKENS["outline_variant"],
        error_background_fill=TOKENS["error_container"],
        error_text_color=TOKENS["on_error_container"],
        error_border_color=TOKENS["error"],
    )
    return theme


MATERIAL_THEME = build_theme()


# ─────────────────────────────────────────────────────────────────────────
# Supplemental CSS — Material app bar, cards, chips, tab indicator,
# progress bars, console log panels, focus rings. Targets elem_classes
# this app assigns itself (reliable across Gradio versions) plus a small
# set of Gradio's own long-stable structural selectors (.tabs/.tab-nav).
# ─────────────────────────────────────────────────────────────────────────
MATERIAL_CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;600;700&family=Roboto+Mono:wght@400;500;600&display=swap');

:root {{
  --m3-primary: {TOKENS["primary"]};
  --m3-on-primary: {TOKENS["on_primary"]};
  --m3-primary-container: {TOKENS["primary_container"]};
  --m3-on-primary-container: {TOKENS["on_primary_container"]};
  --m3-secondary: {TOKENS["secondary"]};
  --m3-secondary-container: {TOKENS["secondary_container"]};
  --m3-on-secondary-container: {TOKENS["on_secondary_container"]};
  --m3-surface: {TOKENS["surface"]};
  --m3-surface-container-low: {TOKENS["surface_container_low"]};
  --m3-surface-container: {TOKENS["surface_container"]};
  --m3-surface-container-high: {TOKENS["surface_container_high"]};
  --m3-surface-container-highest: {TOKENS["surface_container_highest"]};
  --m3-on-surface: {TOKENS["on_surface"]};
  --m3-on-surface-variant: {TOKENS["on_surface_variant"]};
  --m3-outline: {TOKENS["outline"]};
  --m3-outline-variant: {TOKENS["outline_variant"]};
  --m3-success: {TOKENS["success"]};
  --m3-success-container: {TOKENS["success_container"]};
  --m3-on-success-container: {TOKENS["on_success_container"]};
  --m3-error: {TOKENS["error"]};
  --m3-error-container: {TOKENS["error_container"]};
  --m3-on-error-container: {TOKENS["on_error_container"]};
  --m3-el1: 0 1px 2px 0 rgba(34,18,82,0.08), 0 1px 3px 1px rgba(34,18,82,0.08);
  --m3-el2: 0 1px 2px 0 rgba(34,18,82,0.10), 0 2px 6px 2px rgba(34,18,82,0.10);
  --m3-el3: 0 1px 3px 0 rgba(34,18,82,0.12), 0 4px 8px 3px rgba(34,18,82,0.12);
}}

* {{ scroll-behavior: smooth; }}

body, .gradio-container {{
  background: var(--m3-surface) !important;
}}

.gradio-container {{
  max-width: 1180px !important;
  font-feature-settings: "cv02","cv03","cv04","cv11";
}}

/* Visible, on-brand focus ring for keyboard navigation (a11y) */
button:focus-visible, input:focus-visible, textarea:focus-visible,
[role="button"]:focus-visible, .tab-nav button:focus-visible {{
  outline: 3px solid var(--m3-primary) !important;
  outline-offset: 2px !important;
  border-radius: 8px;
}}
@media (prefers-reduced-motion: reduce) {{
  * {{ transition: none !important; animation: none !important; scroll-behavior: auto !important; }}
}}

/* ── App bar ─────────────────────────────────────────────────────── */
.m3-appbar {{
  display: flex;
  align-items: center;
  gap: 16px;
  background: linear-gradient(135deg, var(--m3-primary) 0%, #5533C4 100%);
  color: var(--m3-on-primary);
  border-radius: 24px;
  padding: 22px 28px;
  margin-bottom: 20px;
  box-shadow: var(--m3-el2);
}}
.m3-appbar-icon {{
  width: 52px; height: 52px; flex-shrink: 0;
  border-radius: 16px;
  background: rgba(255,255,255,0.16);
  display: flex; align-items: center; justify-content: center;
  font-size: 28px;
}}
.m3-appbar-title {{
  font-size: 22px; font-weight: 700; line-height: 1.25; margin: 0;
  letter-spacing: 0.1px;
}}
.m3-appbar-subtitle {{
  font-size: 13.5px; font-weight: 400; margin: 2px 0 0 0;
  color: rgba(255,255,255,0.86);
}}
.m3-chip {{
  display: inline-flex; align-items: center; gap: 6px;
  background: rgba(255,255,255,0.16);
  color: var(--m3-on-primary);
  font-size: 12px; font-weight: 500;
  padding: 5px 12px; border-radius: 999px;
  font-family: 'Roboto Mono', ui-monospace, monospace;
  white-space: nowrap;
}}

/* ── Section cards ───────────────────────────────────────────────── */
.m3-card {{
  background: var(--m3-surface-container-low) !important;
  border-radius: 20px !important;
  padding: 18px 20px !important;
  box-shadow: var(--m3-el1);
  margin-bottom: 16px;
  border: 1px solid var(--m3-outline-variant);
}}
.m3-card-flush {{
  background: transparent !important;
  box-shadow: none !important;
  border: none !important;
  padding: 0 !important;
}}
.m3-card-title {{
  font-size: 11px !important;
  font-weight: 700 !important;
  letter-spacing: 0.8px !important;
  text-transform: uppercase;
  color: var(--m3-primary) !important;
  margin: 0 0 10px 0 !important;
}}
.m3-card-title p {{ margin: 0 !important; color: var(--m3-primary) !important;
  font-size: 11px !important; font-weight: 700 !important; letter-spacing: 0.8px !important; }}

/* ── Tabs — Material 3 primary tabs (underline indicator) ───────── */
.tabs > .tab-nav {{
  border-bottom: 1px solid var(--m3-outline-variant) !important;
  gap: 4px;
  background: transparent !important;
  margin-bottom: 18px;
}}
.tab-nav button {{
  border: none !important;
  background: transparent !important;
  color: var(--m3-on-surface-variant) !important;
  font-weight: 600 !important;
  font-size: 14px !important;
  padding: 12px 18px !important;
  border-radius: 12px 12px 0 0 !important;
  border-bottom: 3px solid transparent !important;
  transition: background-color 120ms ease, color 120ms ease;
}}
.tab-nav button:hover {{
  background: var(--m3-primary-container) !important;
  color: var(--m3-on-primary-container) !important;
}}
.tab-nav button.selected {{
  color: var(--m3-primary) !important;
  border-bottom: 3px solid var(--m3-primary) !important;
}}

/* ── Buttons — Material "filled" / "tonal" / "text" hierarchy ─────── */
button {{
  font-weight: 600 !important;
  letter-spacing: 0.15px;
}}
.m3-btn-filled, .m3-btn-filled button {{ font-weight: 700 !important; }}
.m3-btn-tonal button, button.m3-btn-tonal {{
  background: var(--m3-primary-container) !important;
  color: var(--m3-on-primary-container) !important;
  box-shadow: none !important;
}}
.m3-btn-tonal button:hover {{ background: #DACCFF !important; }}
.m3-btn-icon button {{
  min-width: 44px !important;
  border-radius: 999px !important;
  background: var(--m3-surface-container-high) !important;
  color: var(--m3-on-surface) !important;
  box-shadow: none !important;
}}
.m3-btn-icon button:hover {{ background: var(--m3-surface-container-highest) !important; }}

/* ── Progress (non-interactive Slider used as a Material linear
      progress bar: hide the drag handle, keep only the filled track) ── */
.m3-progress input[type="range"] {{
  pointer-events: none;
}}
.m3-progress input[type="range"]::-webkit-slider-thumb {{ opacity: 0; }}
.m3-progress input[type="range"]::-moz-range-thumb {{ opacity: 0; }}
.m3-progress .wrap {{ padding-top: 2px !important; }}

/* ── Console-style log panels ───────────────────────────────────── */
.m3-console textarea {{
  background: var(--m3-on-surface) !important;
  color: #E8E2F5 !important;
  font-family: 'Roboto Mono', ui-monospace, monospace !important;
  font-size: 12.5px !important;
  line-height: 1.55 !important;
  border-radius: 14px !important;
  padding: 14px 16px !important;
  border: none !important;
}}
.m3-console label span {{ color: var(--m3-on-surface-variant) !important; }}

/* ── Status banner (used for tab intro copy) ───────────────────────── */
.m3-hint {{
  background: var(--m3-surface-container);
  border-left: 4px solid var(--m3-primary);
  border-radius: 0 14px 14px 0;
  padding: 10px 16px;
  font-size: 13px;
  color: var(--m3-on-surface-variant);
  margin-bottom: 16px;
}}
.m3-hint p {{ margin: 0 !important; }}
.m3-hint strong {{ color: var(--m3-on-surface); }}

/* ── Gallery (image generator results) ─────────────────────────────── */
.m3-gallery {{ border-radius: 20px !important; overflow: hidden; }}
.m3-gallery .thumbnail-item, .m3-gallery img {{ border-radius: 14px !important; }}

/* ── Accordion — Material "surface list" treatment ─────────────────── */
.gr-accordion, .m3-card .label-wrap {{
  border-radius: 14px !important;
}}

/* Dropdowns / textboxes: remove residual hard corners on inner wraps */
.gr-box, .wrap, .form {{ border-radius: 16px !important; }}

footer {{ display: none !important; }}
"""
