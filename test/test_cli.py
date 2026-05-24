import importlib
import sys
import unittest


class TestCliVersion(unittest.TestCase):
    def tearDown(self):
        for module_name in [
            "allTranslate",
            "allTranslate.allTranslate",
            "allTranslate.high_level",
            "allTranslate.doclayout",
        ]:
            sys.modules.pop(module_name, None)

    def test_importing_package_does_not_eagerly_load_translation_pipeline(self):
        pkg = importlib.import_module("allTranslate")

        self.assertEqual(pkg.__version__, "1.9.11")
        self.assertNotIn("allTranslate.high_level", sys.modules)

    def test_version_flag_exits_before_loading_heavy_modules(self):
        cli = importlib.import_module("allTranslate.allTranslate")

        self.assertNotIn("allTranslate.high_level", sys.modules)
        self.assertNotIn("allTranslate.doclayout", sys.modules)

        with self.assertRaises(SystemExit) as exit_context:
            cli.main(["-v"])

        self.assertEqual(exit_context.exception.code, 0)
        self.assertNotIn("allTranslate.high_level", sys.modules)
        self.assertNotIn("allTranslate.doclayout", sys.modules)


if __name__ == "__main__":
    unittest.main()
