
def assignopts(opts, *args):
    """
    Assigns optional arguments to their corresponding values based on the provided options.

    Args:
        opts (list or str): The list of options or a single option flag.
        *args: Variable length arguments representing the optional arguments and their values.

    Returns:
        list: The list of unmatched optional arguments.

    Raises:
        ValueError: If an unrecognized flag is encountered or if the optional arguments and values do not come in pairs.

    Notes:
        - The function checks for flags at the beginning of the `opts` list or the single `opts` flag.
        - If a cell array is passed instead of a list, it is converted to a list.
        - The function supports case-insensitive matching if the 'ignorecase' flag is provided.
        - The function supports exact matching if the 'exact' flag is provided.
        - If multiple matches are found for an option, the function tries to find an exact match. If that fails, the option is ignored.
        - The function assigns the value to the corresponding variable using the *original* option name.
        - The function returns the list of unmatched optional arguments.
        - If there are unmatched optional arguments, a warning message is printed.

    Example:
        opts = ['ignorecase', 'exact']
        args = ['option1', 1, 'option2', 2]
        remain = assignopts(opts, *args)
        print(remain)  # Output: ['option1', 1, 'option2', 2]
    """
    ignorecase = 0
    exact = 0

    # check for flags at the beginning
    while not isinstance(opts, list):
        if opts.lower() == 'ignorecase':
            ignorecase = 1
        elif opts.lower() == 'exact':
            exact = 1
        else:
            raise ValueError('unrecognized flag: ' + opts)

        opts = args[0]
        args = args[1:]

    # if passed cell array instead of list, deal
    if len(args) == 1 and isinstance(args[0], list):
        args = args[0]

    if len(args) % 2 != 0:
        raise ValueError('Optional arguments and values must come in pairs')

    done = [0] * len(args)

    origopts = opts
    if ignorecase:
        opts = [opt.lower() for opt in opts]

    for i in range(0, len(args), 2):
        opt = args[i]
        if ignorecase:
            opt = opt.lower()

        # look for matches
        if exact:
            match = [idx for idx, val in enumerate(opts) if val == opt]
        else:
            match = [idx for idx, val in enumerate(opts) if val.startswith(opt)]

        # if more than one matched, try for an exact match ... if this
        # fails we'll ignore this option.
        if len(match) > 1:
            match = [idx for idx, val in enumerate(opts) if val == opt]

        # if we found a unique match, assign in the corresponding value,
        # using the *original* option name
        if len(match) == 1:
            globals()[origopts[match[0]]] = args[i+1]
            done[i:i+1] = 1

    args = [val for idx, val in enumerate(args) if not done[idx]]
    remain = args

    # Throw warning if unmatched inputs are not retrieved by caller
    if len(args) > 0:
        print('Inputs unassigned:', args[::2])

    return remain