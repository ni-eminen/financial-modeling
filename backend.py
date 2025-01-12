import random
import numpy as np
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from scipy.stats import binom, gamma, multinomial, rv_continuous, rv_discrete
import scipy
from .Operator import Operator
from .BackendConfig import ctx

app = FastAPI()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CreateQuantityPayload(BaseModel):
    quantity_name: str
    operator_name: str
    model: str
    model_params: dict
    categories: list


class GetNewSamplesPayload(BaseModel):
    operator_name: str
    quantity_name: str


class UpdateParametersPayload(BaseModel):
    operator_name: str
    quantity_name: str
    params: dict


class CreateConvolutionPayload(BaseModel):
    quantity1_name: str
    quantity2_name: str
    operation: str
    convolution_name: str
    operator1_name: str
    operator2_name: str

# Pydantic models
class Message(BaseModel):
    role: str
    content: str

class UserMessage(BaseModel):
    message: str

class CollectedData(BaseModel):
    name: str = ""
    email: str = ""
    purpose: str = ""
    preferences: str = ""

class ConversationState(BaseModel):
    messages: List[Message]
    current_step: str
    collected_data: CollectedData
    last_updated: datetime


def get_operator(ctx, name):
    if name == "global":
        return ctx
    for o in ctx.operators:
        if o.name == name:
            return o


@app.get("/")
async def root():
    print('a call made')
    return {"message": "Hello World"}

@app.post("/api/message")
async def handle_message(
    user_message: UserMessage,
    session_id: str = Depends(get_session)
):
    state = sessions[session_id]
    
    # Add user message
    state.messages.append(Message(role="user", content=user_message.message))
    
    try:
        # Get OpenAI response
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[msg.dict() for msg in state.messages]
        )
        
        # Add assistant response
        assistant_message = Message(
            role="assistant",
            content=response.choices[0].message.content
        )
        state.messages.append(assistant_message)
        
        # Update session state
        state.last_updated = datetime.now()
        sessions[session_id] = state
        
        # Return visible messages
        return {
            "messages": [
                msg.dict() for msg in state.messages if msg.role != "system"
            ],
            "currentStep": state.current_step
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# TODO: Now implement in front-end a mechanism that detects when a convolution term is updated and update the
# corresponding convolutions
@app.get("/get-new-samples/", status_code=201)
async def get_new_samples(payload: GetNewSamplesPayload):
    operator_name, quantity_name = payload.operator_name, payload.quantity_name
    operator = get_operator(ctx, operator_name)
    quantity = operator.quantities[quantity_name]

    new_samples = quantity.sample(1000)

    return {"samples": new_samples}


@app.post("/create-operator/{name}", status_code=201)
async def create_operator(name):
    ctx.operators.append(Operator(name=name))
    return {
        "operator_name": name
    }


@app.post("/update-parameters/", status_code=201)
async def update_parameters(payload: UpdateParametersPayload):
    operator_name, quantity_name, params = payload.operator_name, payload.quantity_name, payload.params
    if operator_name == "global":
        operator = ctx
    else:
        operator = ctx.get_operator(name=operator_name)

    quantity = operator.quantities[quantity_name]
    quantity.update_params(params)

    image = quantity.generate_image(update_samples=True)

    return image


@app.post("/create-quantity/", status_code=201)
async def create_quantity(payload: CreateQuantityPayload):
    operator = get_operator(ctx=ctx, name=payload.operator_name) if payload.operator_name != "global" else ctx

    quantity_name = payload.quantity_name
    quantity_model = payload.model
    model_params = payload.model_params
    categories = []
    scipy_quantity = getattr(scipy.stats, quantity_model)

    if quantity_model == 'multinomial':
        categories = payload.categories
        model_params['p'] = [float(p) for p in model_params['p']]
        args_dict_ = model_params.copy()
        args_dict_.pop('categories', None)
        operator.create_quantity(name=quantity_name, sample=scipy_quantity.rvs, kwargs=args_dict_,
                                 domain_type='multinomial', dist_class='multinomial', categories=categories)
    elif isinstance(scipy_quantity, rv_continuous):
        operator.create_quantity(quantity_name, pdf=scipy_quantity.pdf, cdf=scipy_quantity.cdf,
                                 sample=scipy_quantity.rvs, kwargs=model_params, domain_type="continuous")
    elif isinstance(scipy_quantity, rv_discrete):
        operator.create_quantity(quantity_name, pdf=scipy_quantity.pmf, cdf=scipy_quantity.cdf,
                                 sample=scipy_quantity.rvs, kwargs=model_params, domain_type="discrete")

    model = operator.quantities[quantity_name]
    samples = list(model.samples)
    a, b = np.min(samples), np.max(samples)
    if model.domain_type == 'discrete':
        x = list(range(a, b + 1))  # Corrected for discrete domain
    elif model.domain_type == 'continuous':
        x = list(np.linspace(a, b, 1000))
    elif model.domain_type == 'multinomial':
        x = list(range(len(model.categories)))

    pdf_samples = [model.pdf(xi) for xi in x]
    pdf_samples = np.array(pdf_samples) / np.sum(pdf_samples)
    cdf_samples = [model.cdf(xi) for xi in x]

    to_return = {
        "name": quantity_name,
        "type": model.type,
        "operator": operator.name,
        "samples": samples,  # Ensure this is a list
        "pdf_samples": {
            "x": x,
            "y": list(pdf_samples)  # Convert numpy array to list
        },
        "cdf_samples": {
            "x": x,
            "y": list(cdf_samples)  # Ensure CDF samples are lists
        },
        "categories": categories,
        "domain_type": model.domain_type,
        "params": model.kwargs
    }

    return to_return


@app.post("/create-convolution/", status_code=201)
async def create_convolution(payload: CreateConvolutionPayload):
    quantity1_name = payload.quantity1_name
    quantity2_name = payload.quantity2_name
    operation = payload.operation
    convolution_name = payload.convolution_name
    operator1_name = payload.operator1_name
    operator2_name = payload.operator2_name
    operator_convolution = operator1_name == operator2_name

    if operator_convolution:
        operator = get_operator(ctx=ctx, name=payload.operator1_name) if operator1_name != "global" else ctx
        operator.create_convolution(conv_name=convolution_name, quantity1=operator.quantities[quantity1_name],
                                    quantity2=operator.quantities[quantity2_name], operation=operation)
        model = operator.quantities[convolution_name]
        owner = operator.name
    else:
        o1, o2 = get_operator(ctx=ctx, name=operator1_name), get_operator(ctx=ctx, name=operator2_name)
        q1, q2 = o1.quantities[quantity1_name], o2.quantities[quantity2_name]
        ctx.create_convolution(conv_name=convolution_name, quantity1=q1, quantity2=q2, operation=operation)
        model = ctx.quantities[convolution_name]
        owner = "global"

    samples = model.samples
    a, b = np.min(samples), np.max(samples)

    if model.domain_type == 'discrete':
        x = list(range(a, b))
    else:
        x = list(np.linspace(a, b, 1000))

    pdf_samples = [model.pdf(xi) for xi in x]
    pdf_samples = np.array(pdf_samples) / np.sum(pdf_samples)
    cdf_samples = [model.cdf(xi) for xi in x]

    return {
        "name": convolution_name,
        "type": "convolution",
        "operator": owner,
        "samples": samples,
        "pdf_samples": {
            "x": x,
            "y": list(pdf_samples)
        },
        "cdf_samples": {
            "x": x,
            "y": cdf_samples
        },
        "domain_type": model.domain_type,
        "categories": []
    }
