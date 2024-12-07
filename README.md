# Financial modelling for SMBs

This repository contains a FastAPI backend designed for creating and managing statistical models, quantities, and convolutions for financial outcome prediction. The application supports operations with various probability distributions, including binomial, gamma, and categorical models, and provides APIs for sampling, parameter updates, and convolution creation.

## How it works

### Operators

- Operators are the entities that we want to model outcomes for.
- If you want to model a sales team, you create operators that represent the team members.

### Quantities

- Operators can have quantities, which are represented as probability distributions. Example: A salesperson has a quantity, a random variable, that estimates their weekly total sales.
- For a quantity, you can choose the initial model and parameters, that can then be updated later based on new data to create a posterior.

### Convolutions

- An operator may have multiple quantities which together form a more meaningful statistic. This is called a convolution of the quantities.
- For example, a sales person may have a distribution of sales meetings a week ranging from 5 - 20, as well as a success rate of 5 - 10 % per meeting; you can combine these two quantities to create a new distribution of probabilities for successful meetings each week.

## Setting up the front-end

Refer to [this repository](https://github.com/ni-eminen/financial-modeling-front) for the front-end.

## Prerequisites

- Python 3.8+
- [FastAPI](https://fastapi.tiangolo.com/)
- [Scipy](https://www.scipy.org/)
- [Numpy](https://numpy.org/)

## Installation

1. Clone the repository:

   ```bash
   git clone git@github.com:ni-eminen/financial-modeling.git
   cd financial-modeling
   ```

2. Create a virtual environment and activate it:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. Start the FastAPI server:

   ```bash
   uvicorn backend:app --reload
   ```

## API Endpoints

### Base Endpoint (system online check)

- `GET /`
  - **Description:** Returns a "Hello World" message.

### Operators

- `POST /create-operator/{name}`
  - **Description:** Creates a new operator.
  - **Parameters:** `name` - Name of the operator.

### Quantities

- `POST /create-quantity/`

  - **Description:** Creates a new statistical quantity.
  - **Payload:**
    ```json
    {
      "quantity_name": "string",
      "operator_name": "string",
      "model": "string",
      "model_params": { "key": "value" },
      "categories": ["category1", "category2"]
    }
    ```

- `POST /update-parameters/`
  - **Description:** Updates parameters for a specified quantity.
  - **Payload:**
    ```json
    {
      "operator_name": "string",
      "quantity_name": "string",
      "params": { "key": "value" }
    }
    ```

### Sampling

- `GET /get-new-samples/`
  - **Description:** Retrieves new samples for a specified quantity.
  - **Payload:**
    ```json
    {
      "operator_name": "string",
      "quantity_name": "string"
    }
    ```

### Convolutions

- `POST /create-convolution/`
  - **Description:** Creates a convolution of two quantities.
  - **Payload:**
    ```json
    {
      "quantity1_name": "string",
      "quantity2_name": "string",
      "operation": "string",
      "convolution_name": "string",
      "operator1_name": "string",
      "operator2_name": "string"
    }
    ```

## Directory Structure

```
BackendConfig.py        # Backend configuration context
Context.py              # Context management for operators and quantities
Operator.py             # Operator implementation
Distribution.py         # Distribution handling logic
backend.py              # FastAPI application code
README.md               # Documentation
main.py                 # CLI version of the app, no front-end needed
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.
