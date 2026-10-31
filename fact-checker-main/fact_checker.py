import argparse


MAX_QUESTION_LENGTH = 4000
DEFAULT_QUESTION = "What type of mammal lays the biggest eggs?"


def validate_question(question: str) -> str:
    if not isinstance(question, str):
        raise TypeError("Question must be a string.")

    question = question.strip()
    if not question:
        raise ValueError("Question cannot be empty.")
    if len(question) > MAX_QUESTION_LENGTH:
        raise ValueError(
            f"Question must be {MAX_QUESTION_LENGTH} characters or fewer."
        )
    return question


def fact_check(question: str) -> str:
    question = validate_question(question)

    from langchain.chains import LLMChain, SimpleSequentialChain
    from langchain.llms import OpenAI
    from langchain.prompts import PromptTemplate

    llm = OpenAI(temperature=0.7)
    template = """{question}\n\n"""
    prompt_template = PromptTemplate(input_variables=["question"], template=template)
    question_chain = LLMChain(llm=llm, prompt=prompt_template)

    template = """Here is a statement:
    {statement}
    Make a bullet point list of the assumptions you made when producing the above statement.\n\n"""
    prompt_template = PromptTemplate(input_variables=["statement"], template=template)
    assumptions_chain = LLMChain(llm=llm, prompt=prompt_template)

    template = """Here is a bullet point list of assertions:
    {assertions}
    For each assertion, determine whether it is true or false. If it is false, explain why.\n\n"""
    prompt_template = PromptTemplate(input_variables=["assertions"], template=template)
    fact_checker_chain = LLMChain(llm=llm, prompt=prompt_template)

    template = """In light of the above facts, how would you answer the question '{}'""".format(question)
    template = """{facts}\n""" + template
    prompt_template = PromptTemplate(input_variables=["facts"], template=template)
    answer_chain = LLMChain(llm=llm, prompt=prompt_template)

    overall_chain = SimpleSequentialChain(chains=[question_chain, assumptions_chain, fact_checker_chain, answer_chain], verbose=True)

    return overall_chain.run(question)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fact-check a question with an LLM.")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    args = parser.parse_args()

    try:
        question = validate_question(args.question)
    except (TypeError, ValueError) as error:
        parser.error(str(error))

    print(question)
    answer = fact_check(question)
    print(answer)


if __name__ == "__main__":
    main()