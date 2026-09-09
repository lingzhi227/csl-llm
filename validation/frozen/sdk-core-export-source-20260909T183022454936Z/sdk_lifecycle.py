"""Independent cleanup attempts for a caller-owned SDK instance."""


def describe(error):
    return None if error is None else dict(type=type(error).__name__, message=str(error))


def stop_and_record(runner, *, stage, record, primary_error=None, references=()):
    """Attempt every cleanup and receipt even when logging or stop fails.

    Call from finally while the primary exception, if any, is propagating.
    Preserve that exception; otherwise propagate the first cleanup failure.
    Receipt failures are attached as exception notes when another failure is
    already active. This does not mark SDK results numerically accepted.
    """
    errors = []
    actions = [('stage-stop', lambda: stage('stop')), ('runner-stop', runner.stop)]
    actions.extend((f'reference-close-{index}', reference.close)
                   for index, reference in enumerate(references))
    for operation, action in actions:
        try:
            action()
        except BaseException as error:
            errors.append((operation, error))
    try:
        record(dict(primary_error=describe(primary_error),
            cleanup_errors=[dict(operation=operation, **describe(error)) for operation, error in errors]))
    except BaseException as error:
        errors.append(('evidence-write', error))
    if primary_error is not None:
        for operation, error in errors:
            primary_error.add_note(f'{operation}: {type(error).__name__}: {error}')
    elif errors:
        first = errors[0][1]
        for operation, error in errors[1:]:
            first.add_note(f'{operation}: {type(error).__name__}: {error}')
        raise first
