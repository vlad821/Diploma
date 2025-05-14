from dash import html

# Dark theme with improved transitions
def dark_theme():
    return {
        "backgroundColor": "#121212",
        "color": "#E0E0E0",
        "border": "1px solid #333",
        "padding": "16px",
        "borderRadius": "10px",
        "fontFamily": "Segoe UI, sans-serif",
        "boxShadow": "0 4px 12px rgba(0, 0, 0, 0.4)",
        "transition": "all 0.3s ease-in-out",
        "margin": "10px",
    }

# Light theme with improved transitions
def light_theme():
    return {
        "backgroundColor": "#F9F9F9",
        "color": "#212121",
        "border": "1px solid #DDD",
        "padding": "16px",
        "borderRadius": "10px",
        "fontFamily": "Segoe UI, sans-serif",
        "boxShadow": "0 4px 12px rgba(0, 0, 0, 0.1)",
        "transition": "all 0.3s ease-in-out",
        "margin": "10px",
    }

# Button style with hover effects
def button_style(is_dark=True):
    return {
        "backgroundColor": "#1E88E5" if is_dark else "#1976D2",
        "color": "#FFFFFF",
        "padding": "10px 20px",
        "border": "none",
        "borderRadius": "5px",
        "cursor": "pointer",
        "fontWeight": "bold",
        "marginTop": "10px",
        "boxShadow": "0 2px 4px rgba(0, 0, 0, 0.3)" if is_dark else "0 2px 4px rgba(0, 0, 0, 0.15)",
        "transition": "all 0.3s ease-in-out",
    }

# Input style with smoother transition effects
def input_style(is_dark=True):
    return {
        "backgroundColor": "#2A2A2A" if is_dark else "#FFFFFF",
        "color": "#FFFFFF" if is_dark else "#000000",
        "padding": "8px",
        "border": "1px solid #555" if is_dark else "1px solid #CCC",
        "borderRadius": "5px",
        "width": "100%",
        "marginTop": "8px",
        "transition": "all 0.3s ease-in-out",
    }

# Toggleable theme selection
def set_theme(is_dark_mode=True):
    return dark_theme() if is_dark_mode else light_theme()

# Toggle button for dark/light mode
def toggle_theme_button(is_dark_mode=True):
    return html.Button(
        "Switch Theme",
        style=button_style(is_dark_mode),
        id="toggle-theme-btn",
    )
