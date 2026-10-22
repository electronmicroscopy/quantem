# quantem

## Tests

The test suite protects the code against regressions for the next person who edits it. CI runs
every test on every push to every pull request, so each test adds to the CI time of every PR.

- Before writing a new test, find the existing test for that code and extend it. Add a new
  test file only for a new module.
- Add a test for a bug fix (a test that fails before the fix) or for a new code path.
  Refactors, renames, argument forwarding, default values, getters, reprs and plotting do not
  need new tests.
- Test each code path once. Do not parametrize over values that take the same branch, and do
  not take cartesian products of parameters. When several inexpensive cases share one code
  path, loop over them inside one test and name the case in the assert message.
- Every assert must be able to fail. Do not assert only `is not None`, a type, or a shape that
  is fixed by construction, do not compare a result with itself, and do not use `try/except`
  or `skip` in a way that hides a failure.
- Use the smallest problem that exercises the code path, such as small arrays, few iterations
  and a coarse orientation plan, and build expensive inputs once in a module-scoped fixture.
- CI runs tests marked `@pytest.mark.slow` only on pull requests into `main`, once per release,
  so mark a test as slow only when checking it at each release is enough.
- When a test duplicates another test, delete it.
