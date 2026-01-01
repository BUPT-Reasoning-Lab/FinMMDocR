prompt_generalize="""
You will receive a financial research report. Based on the content of this report, design 3 English graduate-level questions that are as complex as possible. The difficulty of each question should derive from the following three aspects:

1. **Numerical Calculation Complexity**: The calculation process must involve multiple steps and should not be solvable with just a few simple calculations.  
2. **Conceptual Understanding**: Each question must be set in a financial context and should assess the understanding and application of financial terminology and concepts.  
3. **Data Extraction Difficulty**: The numerical data required to solve the problem must be retrieved from multiple parts of the report. It should not all be found on a single page, within a single chart, or in one paragraph. You are encouraged to
extract clear data from the chart for problem solving or extract data from the later part of the document for problem solving (to ensure difficulty). Without fabricating inaccurate data from charts without clear data, data sources should include tables, images, and charts as much as possible.

Each question must have **only one numerical answer**. The output must be a **plain number**—**no units, no percent signs**. The question must specify the required units and number of significant digits.

You are allowed to **create reasonable assumptions/hypothetical scenarios** or through other means to enhance complexity. For example, you may introduce cost estimation scenarios by manually setting additional values, or create forecasting questions with assumptions like linear trends, etc.

Extracting data from research reports is part of the difficulty of the topic. Therefore, in the **question text**, you **must not mention** the specific formulas being tested or the sources/numbers used for the calculations.

In addition, provide a **detailed solution for each question**, which must include:

- An explanation that clearly states **where the data came from** (e.g., page number, table/chart, or paragraph reference in the report).
- A **Python code snippet** that solves the problem, following the format below:

python
def solution():
# Define variables with their values
revenue = 600000
avg_account_receivable = 50000

# Do financial calculation
receivables_turnover = revenue / avg_account_receivable
answer = 365 / receivables_turnover

# Return final result
return answer


### Example of a High-Quality Question:
Assume that in 2026, VLLC (Weilan Lithium Core) continues to operate based on the forecasted revenue and gross margin data provided in the report. However, a new business structure emerges within the battery segment: the Backup Battery Unit (BBU) accounts for 30% of this segment's revenue, and its gross margin is 10 percentage points higher than the overall battery segment’s gross margin provided in the report. The gross margin for the power tool battery subsegment remains unchanged. All other business segments (LED, metal logistics, and “Others”) maintain the forecasted revenue and gross margin levels from the report. Under these assumptions, calculate the company’s total gross profit in 2026** (round to two decimal places, unit: 100 million yuan).
"""