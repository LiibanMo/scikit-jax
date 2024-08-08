import unittest

if __name__ == "__main__":
    # Discover and run tests from the 'tests' directory
    unittest.TextTestRunner().run(unittest.defaultTestLoader.discover("tests"))
