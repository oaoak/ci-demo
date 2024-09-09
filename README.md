Continuous Integration Demo
===========================

This project demonstrates use of Github Actions to build and test a Python project.  

## Automatic Testing

1. Create a repository on Github containing this starter code
2. Add a Github Action to run unit tests.
3. Pull from Github so your local repositry has the action (update your local repo).
4. The Github Action should initially fail.
   - Fix the code and tests locally until they pass on your machine.
   - Push the fix to Github.
   - Observe how Github automatically reruns your Action and updates the badge.
5. Add a Github "Badge" showing test status to this README.


## Flake8

1. Create a separate Github Action for flake8 and flake8-docstrings.
2. Modify the action so that it **does** exit with a non-zero exit code when errors are found.
3. Pull the Github repo, again.
4. Fix the style problems until your code passes flake8.
5. Push to Github.

## Code Coverage (Not Done Yet)

Use Github to perform code coverage and generate a report.
This will help you find places where you need to improve your tests.

1. Modify your Github Action for unit tests to run tests with code coverage
2. Send a coverage report (as xml) to <https://codecov.io>
3. Add a Codecov badge to this README


