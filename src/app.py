import app_covid2_README
import app_covid2

import streamlit as st

# CUSTOM FOOTER BEGIN
from htbuilder import HtmlElement, div, ul, li, br, hr, a, p, img, styles, classes, fonts
from htbuilder.units import percent, px
# from htbuilder.funcs import rgba, rgb

def image(src_as_string, **style):
    return img(src=src_as_string, style=styles(**style))
def link(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(**style))(text)
def layout(*args):
    style = """
    <style>
      # MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
     .stApp { bottom: 105px;  }
    </style>
    """
    style_div = styles(
        position="fixed",
        left=0,
        bottom=0,
        margin=px(0, 0, 0, 0),
        width=percent(100),
        color="black",
        text_align="center",
        height="auto",
        opacity=0.75
    )
    style_hr = styles(
        display="block",
        margin=px(8, 8, 0, 0),
        border_style="inset",
        border_width=px(1)
    )
    body = p()
    foot = div(
        style=style_div
    )(
        hr(
            style=style_hr
        ),
        body
    )
    st.markdown(style, unsafe_allow_html=True)
    for arg in args:
        if isinstance(arg, str):
            body(arg)
        elif isinstance(arg, HtmlElement):
            body(arg)
    st.markdown(str(foot), unsafe_allow_html=True)

def footer():
    myargs = [
        "Made in ",
        image('https://avatars3.githubusercontent.com/u/45109972?s=400&v=4',
              width=px(25), height=px(25)),
        " with ❤️ by ",
        link("https://github.com/fontanads", "@fontanads"),
        br(),
        link("https://github.com/fontanads/bootcamp_dsa_2021/", image('https://github.com/fontanads/bootcamp_dsa_2021/raw/main/pics/GitHub-Mark/PNG/GitHub-Mark-32px.png')),
        link("https://github.com/fontanads/bootcamp_dsa_2021/", "Projeto no GitHub")
    ]
    layout(*myargs)
# CUSTOM FOOTER END

footer()
PAGES = {
    "HEY, LEIA-ME! :)": app_covid2_README,
    "App": app_covid2
}
st.sidebar.title('MENU')
selection = st.sidebar.radio("IR PARA", list(PAGES.keys()))
page = PAGES[selection]
page.main()

