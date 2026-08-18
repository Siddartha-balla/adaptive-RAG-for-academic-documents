---
name: Streamlit CSS selectors (v1.59)
description: Tested selectors for overriding Streamlit internals in v1.59.2
---

## Chat
- `[data-testid="stChatMessage"]` — message bubble wrapper
- `[data-testid="stChatMessageContent"]` — inner text area
- `[data-testid="chatAvatarIcon-user"]` — user avatar circle
- `[data-testid="chatAvatarIcon-assistant"]` — assistant avatar circle
- `[data-testid="stChatInput"]` — input bar wrapper
- `[data-testid="stChatInputTextArea"]` — textarea inside input bar

## Layout
- `.stApp` — full page background
- `[data-testid="stSidebar"]` — sidebar panel
- `.main .block-container` — center content wrapper (use max-width + margin:auto)
- `[data-testid="stSidebarContent"]` — inner sidebar content

## Controls
- `[data-testid="stBaseButton-primary"]` — primary button (not `.stButton > button[kind="primary"]`)
- `[data-testid="stMetric"]`, `[data-testid="stMetricLabel"]`, `[data-testid="stMetricValue"]`
- `[data-testid="stTabs"]`, `[data-testid="stTabsBar"]`, `button[data-testid="stTab"]`
- `[data-testid="stProgress"] > div > div` — progress bar fill
- `[data-testid="stFileUploader"]`
- `[data-testid="stAudioInput"]`
- `[data-testid="stExpander"]`

## Note
- `st.button()` does NOT accept `label_visibility` — that's only for input widgets
- `:has()` CSS selector works in modern Chromium (Replit preview uses it)
- MathJax/Prism must be injected into parent frame via `st.components.v1.html()` + `window.parent.document`
- Streamlit strips `<script>` tags from `st.markdown(unsafe_allow_html=True)`
