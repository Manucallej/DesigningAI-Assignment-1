# Assignment 1 Report

*Delete this italic guidance as you fill in each section. You'll be asked to
defend any of this without your code in front of you — write only what you
can actually explain.*

- **Name**: Manuela Calle
- **Student ID**: 18754
- **Email**: Mcalle.ieu2022@student.ie.edu
- **Group**: BBADBA 5A 

## Dataset

*What is it, where did you get it, what does one row represent, how many
rows/columns, and why did you pick it.*

It's a hotel booking dataset, I got it from kaggle: https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand
One row represents one hotel booking. This is the shape of the dataset: rows, columns: (119390, 32). 
I picked it because it is of personal interest to me knowing what are the factors that go into hotel cancellations. I'm incredible invested in the hotel and hospitality industry, so learning about cancellation rates and overbooking appropriately is important to my field. 

## Business / real-life framing

*The hypothetical scenario this model serves, and what that scenario implies
for how you built the pipeline — target definition, whether a time-based
split was necessary and why (or why not), which metric should drive the
decision threshold and why.*
The hypothetical scenario this model serves is that it helps hotels predict cancellations so they can know up to which rate they should overbook rooms and what consequent actions they should take. For example, overbook or require a deposit. A time split was necessary because there is a time progression and this is time-dependent. A random split would let the "future" data leak into the training process. We're predicting "is_canceled". In this project, the variable that guides that is the expected business cost led by the equation in the code FP × cost_FP + FN × cost_FN (cost_FP and cost_FN are hard_coded in the code as well). The idea is to modify the threshold in such way that the business cost is minimized. 

## Data preparation & feature engineering

*What you engineered and why, and any data-quality decisions you made along
the way — e.g. "segment X had defective data, so I excluded it and used a
population-average default for scope Y at inference time; the impact of
that choice is Z."*
Replaced null values in numerical variables with the Median through simpleimputer. Replaced null values in categorical variables with the mode through simpleimputer. Scaled numerical variables through standardscaler because logistic regression would assign importance differently if they weren't scaled, and encoded categorical variables through a OneHot encoder, because a logistic regression can't handle categories. Turned the months into numerical by creating a dictionary (for the time split, they are used categorically later when fed into the model). Deleted variables that leaked or directly predicted the cancellation because they are information that we don't have at booking time such as "reservation_status", "reservation_status_date", and "assigned_room_type". I also removed impossible rows (negative ADR, ADR above 1000, zero-guest bookings) and built flag features (has_agent, has_company, is_family, total_nights, total_guests). Additionally, and importantly, since I found that country was an important variable but we couldn't one hot encode every single country, I kept the top 10 countries, which made up 84,43% of my dataset, and labeled the rest, "other". 


## Modeling: three implementations, one model

*Which model (linear or logistic regression) and why. A results table
comparing scikit-learn, the manual PyTorch loop, and the standard
torch.nn.Module/torch.optim workflow, on the same test set, against the
naive baseline. Do the three agree? If not, why not?*
We're using logistic regression because we're predicting a binary output: either they cancel or they don't cancel the reservation. 
model,accuracy,auc
baseline (majority),0.6049,0.5
sklearn,0.7415,0.8001
manual pytorch,0.7428,0.7993
standard pytorch,0.7418,0.8002



## Limitations & next steps

*Real limitations you found, and concretely how you'd address each one with
more time or data — not generic hedging.*
The costs input to the model are an estimation, and they will affect the actual economic loss that the model estimates in total. Additionally, model-wise, a logistic regression cannot capture interactions between variables, it only captures linear relationships. Another big limitation is that an important column: "Country", which will heavily affect Purchasing Power Parity and therefore room price and other important variables, is dropped. This is because of the limitation of a logistic regression with categorical variables. Since there are 178 options of countries in the dataset, there would be 178 new columns, one-hot encoded, added to the dataset at the moment of training. I personally wouldn't use a model that cannot capture interactions between variables and that has limitations for the variables and needs explicit feature engineering such as scaling to be able to digest some of the data. Additionally, I'd research more on the context and business case on how much money is actually being lost per FP and FN. 

## Generative AI use disclosure

*Per the syllabus AI Policy: what you used and how, or "no AI content used."*
I did use Claude, as assigned by the course, to create the code. However, I went through absolutely all the content, and checked it was correct. I manually copy-pasted the code to make sure I wasn't creating an incorrect pipeline and cross-checked with the instructions constantly. Additionally, I DID NOT use it for any ideation purposes. The ideas are genuinely mine and hotels are literally my thesis topic. I researched everything, created the report on my own, and respected the limitations of the assignment. 