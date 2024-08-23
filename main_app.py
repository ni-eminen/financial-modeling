import numpy as np
import pandas as pd
import streamlit as st
import Operator
from scipy.stats import gamma, binom
from Operator import Operator
from Distribution import Distribution


distributions = [
    "gamma", "binom"
]

@st.dialog("Create operator quantity")
def operator_actions(operator):
    quantity_name = st.text_input('quantity name')
    quantity_model = st.selectbox("quantity distribution", distributions)

    if quantity_model == 'binom':
        model = binom
        args = ['n', 'p']
    elif "gamma":
        model = gamma
        args = ['a', 'scale']

    args_dict = {}
    for arg in args:
        val = st.number_input(f"{arg} value")
        args_dict[arg] = val

    if st.button('Create quantity'):
        if quantity_model == 'binom':
            args_dict['n'] = int(args_dict['n'])
            operator.create_quantity(name=quantity_name, pdf=binom.pmf, cdf=binom.cdf,
                                     sample=binom.rvs, kwargs=args_dict, domain_type='discrete')
        if quantity_model == 'gamma':
            operator.create_quantity(name=quantity_name, pdf=gamma.pdf, cdf=gamma.cdf,
                                     sample=gamma.rvs, kwargs=args_dict, domain_type='continuous')
        st.session_state.quantity_dialog_open = False
        st.rerun()

@st.dialog("Operator actions")
def placeholder(operator: Operator):
    st.write("What would you like to do?")
    if st.button("Create quantity"):
        create_operator_quantity(operator)

@st.dialog("Create operator")
def create_operator():
    name = st.text_input("Name of operator")
    st.session_state.operators.append(Operator(name))
    if st.button("Create"):
        st.rerun()


if 'operators' not in st.session_state:
    st.session_state.operators = []

if 'selected_operator' not in st.session_state:
    st.session_state.selected_operator = None

st.title("ŷhat")

utilities_col, quantities_col = st.columns([2, 3])
utilities_col.write("# Utilities")

# Check for button clicks before looping through operators
if utilities_col.button("Create operator"):
    create_operator()

# Now, create buttons for each operator and track the clicked one
for i, o in enumerate(st.session_state.operators):
    if utilities_col.button(o.name, key=f"operator_{i}"):
        st.session_state.selected_operator = i

# Conditionally render a modal-like expander when an operator is clicked
if st.session_state.selected_operator is not None:
    selected_operator = st.session_state.operators[st.session_state.selected_operator]
    if 'quantity_dialog_open' not in st.session_state:
        st.session_state.quantity_dialog_open = True

    if st.session_state.quantity_dialog_open:
        operator_actions(selected_operator)

quantities_col.write("# Quantities")
for o in st.session_state.operators:
    for quantity in o.quantities.keys():
        a, b = np.min(o.quantities[quantity].samples), np.max(o.quantities[quantity].samples)
        quantities_col.write(f'{o.name} - {o.quantities[quantity]}')
        if o.quantities[quantity].domain_type == 'discrete':
            x = list(range(a, b))
        else:
            x = list(np.linspace(a, b, 10000))

        y = [o.quantities[quantity].pdf(x_) for x_ in x]

        data = pd.DataFrame({
            'x': x,
            'y': y
        })

        quantities_col.write(f"## {o.name} - {o.quantities[quantity].name}")
        quantities_col.line_chart(data.set_index('x'), x_label='$', y_label='likelihood')
