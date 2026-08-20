# Contributing

Two things worth knowing before your first commit. `make help` lists the rest, and
[AGENTS.md](AGENTS.md) is the design reasoning behind the boundaries the checks enforce.

## Git hooks

The hooks are checked in, under [`.githooks/`](.githooks/). Point git at them once per clone:

```bash
make hooks    # or: git config core.hooksPath .githooks
```

Nothing does this for you: git ignores a hooks path a repository declares, because a repository
that could install its own hooks could run code on clone.

**pre-commit** repairs what needs no judgement and stages the result — `ruff format` over the
staged Python, and a rebuild of the chat panel's knowledge bundle
(`visualizer-web/knowledge/bundle.generated.ts`) when you stage anything it is compiled from. That
bundle is committed and is what the deployed site serves, so forgetting it ships a bot describing
the previous release.

**pre-push** runs CI's static half, scoped to the paths the push touches: ruff lint and format,
both pyright configs, the lint-config parity check, the weight-free guard tests (packaging,
published benchmark tables, vocabulary boundary, doc fences) and the visualizer's eslint, tsc and
link checks. It refuses the push once, with the whole list. The model suites are not run — they
want weights, a network and minutes, and `make test` owns them.

Skip either with `IE_SKIP_HOOKS=1`, which says in the terminal that it skipped, or with
`--no-verify`, which does not.

## License

By contributing, you agree that your contributions are licensed under the
[Apache License 2.0](LICENSE), the same license that covers this project.
