from tools.mcq_parser_tool import mcqs_parser_tool

def main():
    raw_text = """Here are 5 multiple-choice questions based on the provided financial document, designed to test understanding rather than just memorization:

**Question 1:**  What is CVS Group plc's primary focus in terms of animal healthcare?

A) Research and Development 
B) Providing high-quality care to animals
C) Online retail business for pet products.
D) Buying and selling veterinary consumable supplies.

**Correct Answer:** B) Providing high-quality care to animals 
**Explanation:** The document repeatedly emphasizes the company's commitment to providing "the best possible care" to animals, both in their practices and through services like the Healthy Pet Club.

**Question 2:**  In what year did CVS Group plc begin its Australian veterinary service expansion?

A) 2021
B) 2022
C) 2023
D) 2024

**Correct Answer:** B) 2022
**Explanation:** The document states that "CVS Group plc entered the Australian veterinary services market in 2022" which is a significant undertaking.


**Question 3:**  What was the primary reason for the decrease in profit before tax in 2023?

A) Increased acquisition costs
B) Disruption from the cyber incident
C) Higher depreciation and amortization expenses
D) A combination of factors including business combination costs, finance expense, and increased investment in people.

**Correct Answer:** D) A combination of factors including business combination costs, finance expense, and increased investment in people. 
**Explanation:** The document outlines that the profit before tax decreased due to several contributing factors: business combination costs, finance expenses, and increased investment in people.


**Question 4:**  What is one example of CVS Group plc's commitment to preventative healthcare?

A) Online retail business "Animed Direct"
B) Horse Health Program (HHP)
C) The Healthy Pet Club (HPC) 
D) Vet Direct, a buying group for veterinary supplies.

**Correct Answer:** C) The Healthy Pet Club (HPC)
**Explanation:**  The document highlights the company's commitment to preventative healthcare through schemes like the "Healthy Pet Club" which provides pet owners with access to preventative care and services. 


**Question 5:**  What was the primary reason for the increase in leverage for CVS Group plc?

A) Increased investment in acquiring new practices
B) Expansion into new markets, such as Australia
C) Higher demand for veterinary services.
D) Increased debt financing for acquisitions and expansion projects. 

**Correct Answer:** D) Increased debt financing for acquisitions and expansion projects. 
**Explanation:**  The document mentions that leverage increased to 1.54x due to the company's investment in acquiring new practices and expanding operations.   
"""

    result = mcqs_parser_tool._run(raw_text=raw_text)
    print("Result:", result)
    print("Type:", type(result))

if __name__ == "__main__":
    main()

