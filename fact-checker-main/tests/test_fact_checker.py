import unittest

from fact_checker import MAX_QUESTION_LENGTH, validate_question


class ValidateQuestionTests(unittest.TestCase):
    def test_strips_surrounding_whitespace(self) -> None:
        self.assertEqual(validate_question("  Is this true?  "), "Is this true?")

    def test_rejects_non_string_questions(self) -> None:
        with self.assertRaisesRegex(TypeError, "must be a string"):
            validate_question(None)

    def test_rejects_empty_questions(self) -> None:
        for question in ("", " \t\n"):
            with self.subTest(question=question):
                with self.assertRaisesRegex(ValueError, "cannot be empty"):
                    validate_question(question)

    def test_enforces_maximum_question_length(self) -> None:
        self.assertEqual(validate_question("x" * MAX_QUESTION_LENGTH), "x" * MAX_QUESTION_LENGTH)
        with self.assertRaisesRegex(ValueError, "4000 characters or fewer"):
            validate_question("x" * (MAX_QUESTION_LENGTH + 1))


if __name__ == "__main__":
    unittest.main()
