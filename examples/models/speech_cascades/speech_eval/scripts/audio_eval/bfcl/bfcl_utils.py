"""Utility functions for BFCL dataset"""

import ast
import json
import os


def sort_key(entry):
    """
    Index comes in two forms: TestCategory_Index or TestCategory_Index-FuncDocSubIndex-PromptSubIndex; both 0-indexed.

    TestCategory_Index: For example, `simple_20` means the 21st entry in the `simple` test category.

    TestCategory_Index-FuncDocSubIndex-PromptSubIndex is used when there are multiple prompts for a single function doc; this only happens in the live dataset.
    FuncDocSubIndex increments for each unique function doc.
    PromptSubIndex is per function doc. It resets to 0 for each function doc.
        For example, `live_simple_19-3-15` means the 20th entry in the `live_simple` test category.
        This entry has the 4th unique function doc and the 16th prompt for that function doc (there are at least 15 other prompts for this same function doc in this category).

    In either case, the universal index is enough to sort the entries.
    """
    parts = entry["id"].rsplit("_", 1)
    test_category, index = parts[0], parts[1]
    # This handles the case where the index is in the form TestCategory_Index-FuncDocSubIndex-PromptSubIndex
    if "-" in index:
        index = index.split("-")[0]
    return (test_category, int(index))


def parse_javascript_function_call(source_code):
    import tree_sitter_javascript
    from tree_sitter import Language, Parser

    JS_LANGUAGE = Language(tree_sitter_javascript.language(), "javascript")

    parser = Parser()
    parser.set_language(JS_LANGUAGE)
    # Parse the source code
    tree = parser.parse(bytes(source_code, "utf8"))
    root_node = tree.root_node
    sexp_result = root_node.sexp()
    if "ERROR" in sexp_result:
        raise Exception("Error js parsing the source code.")

    # Function to recursively extract argument details
    def extract_arguments(node):
        args = {}
        for child in node.children:
            if child.type == "assignment_expression":
                # Extract left (name) and right (value) parts of the assignment
                name = child.children[0].text.decode("utf-8")
                value = child.children[2].text.decode("utf-8")
                if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
                    value = value[1:-1]  # Trim the quotation marks
                if name in args:
                    if not isinstance(args[name], list):
                        args[name] = [args[name]]
                    args[name].append(value)
                else:
                    args[name] = value

            elif child.type == "identifier" or child.type == "true":
                # Handle non-named arguments and boolean values
                value = child.text.decode("utf-8")
                if None in args:
                    if not isinstance(args[None], list):
                        args[None] = [args[None]]
                    args[None].append(value)
                else:
                    args[None] = value
        return args

    # Find the function call and extract its name and arguments
    if root_node.type == "program":
        for child in root_node.children:
            if child.type == "expression_statement":
                for sub_child in child.children:
                    if sub_child.type == "call_expression":
                        function_name = sub_child.children[0].text.decode("utf8")
                        arguments_node = sub_child.children[1]
                        parameters = extract_arguments(arguments_node)
                        for key, value in parameters.items():
                            if isinstance(value, list):
                                raise Exception("Error: Multiple arguments with the same name are not supported.")
                        result = [{function_name: parameters}]
                        return result


def parse_java_function_call(source_code):
    import tree_sitter_java
    from tree_sitter import Language, Parser

    JAVA_LANGUAGE = Language(tree_sitter_java.language(), "java")

    parser = Parser()
    parser.set_language(JAVA_LANGUAGE)

    tree = parser.parse(bytes(source_code, "utf8"))
    root_node = tree.root_node
    sexp_result = root_node.sexp()

    if "ERROR" in sexp_result:
        raise Exception("Error parsing java the source code.")

    def get_text(node):
        """Returns the text represented by the node."""
        return source_code[node.start_byte : node.end_byte]

    def traverse_node(node, nested=False):
        if node.type == "string_literal":
            if nested:
                return get_text(node)
            # Strip surrounding quotes from string literals
            return get_text(node)[1:-1]
        elif node.type == "character_literal":
            if nested:
                return get_text(node)
            # Strip surrounding single quotes from character literals
            return get_text(node)[1:-1]
        """Traverse the node to collect texts for complex structures."""
        if node.type in [
            "identifier",
            "class_literal",
            "type_identifier",
            "method_invocation",
        ]:
            return get_text(node)
        elif node.type == "array_creation_expression":
            # Handle array creation expression specifically
            type_node = node.child_by_field_name("type")
            value_node = node.child_by_field_name("value")
            type_text = traverse_node(type_node, True)
            value_text = traverse_node(value_node, True)
            return f"new {type_text}[]{value_text}"
        elif node.type == "object_creation_expression":
            # Handle object creation expression specifically
            type_node = node.child_by_field_name("type")
            arguments_node = node.child_by_field_name("arguments")
            type_text = traverse_node(type_node, True)
            if arguments_node:
                # Process each argument carefully, avoiding unnecessary punctuation
                argument_texts = []
                for child in arguments_node.children:
                    if child.type not in [
                        ",",
                        "(",
                        ")",
                    ]:  # Exclude commas and parentheses
                        argument_text = traverse_node(child, True)
                        argument_texts.append(argument_text)
                arguments_text = ", ".join(argument_texts)
                return f"new {type_text}({arguments_text})"
            else:
                return f"new {type_text}()"
        elif node.type == "set":
            # Handling sets specifically
            items = [traverse_node(n, True) for n in node.children if n.type not in [",", "set"]]
            return "{" + ", ".join(items) + "}"

        elif node.child_count > 0:
            return "".join(traverse_node(child, True) for child in node.children)
        else:
            return get_text(node)

    def extract_arguments(args_node):
        arguments = {}
        for child in args_node.children:
            if child.type == "assignment_expression":
                # For named parameters
                name_node, value_node = child.children[0], child.children[2]
                name = get_text(name_node)
                value = traverse_node(value_node)
                if name in arguments:
                    if not isinstance(arguments[name], list):
                        arguments[name] = [arguments[name]]
                    arguments[name].append(value)
                else:
                    arguments[name] = value
                # arguments.append({'name': name, 'value': value})
            elif child.type in ["identifier", "class_literal", "set"]:
                # For unnamed parameters and handling sets
                value = traverse_node(child)
                if None in arguments:
                    if not isinstance(arguments[None], list):
                        arguments[None] = [arguments[None]]
                    arguments[None].append(value)
                else:
                    arguments[None] = value
        return arguments

    def traverse(node):
        if node.type == "method_invocation":
            # Extract the function name and its arguments
            method_name = get_text(node.child_by_field_name("name"))
            class_name_node = node.child_by_field_name("object")
            if class_name_node:
                class_name = get_text(class_name_node)
                function_name = f"{class_name}.{method_name}"
            else:
                function_name = method_name
            arguments_node = node.child_by_field_name("arguments")
            if arguments_node:
                arguments = extract_arguments(arguments_node)
                for key, value in arguments.items():
                    if isinstance(value, list):
                        raise Exception("Error: Multiple arguments with the same name are not supported.")
                return [{function_name: arguments}]

        else:
            for child in node.children:
                result = traverse(child)
                if result:
                    return result

    result = traverse(root_node)
    return result if result else {}


def is_multi_turn(test_category):
    return "multi_turn" in test_category


def contain_multi_turn_irrelevance(test_category):
    return "miss_func" in test_category or "miss_param" in test_category


def is_executable(test_category):
    return "exec" in test_category or "rest" in test_category


def is_rest(test_category):
    return "rest" in test_category


def is_relevance_or_irrelevance(test_category):
    return "relevance" in test_category or "irrelevance" in test_category


def is_chatable(test_category):
    return "chatable" in test_category


def is_java(test_category):
    return "java" in test_category


def is_js(test_category):
    return "javascript" in test_category


def is_sql(test_category):
    return "sql" in test_category


def load_file(file_path, sort_by_id=False):
    result = []
    with open(file_path) as f:
        file = f.readlines()
        for line in file:
            result.append(json.loads(line))

    if sort_by_id:
        result.sort(key=sort_key)
    return result


def resolve_ast_call(elem):
    # Handle nested attributes for deeply nested module paths
    func_parts = []
    func_part = elem.func
    while isinstance(func_part, ast.Attribute):
        func_parts.append(func_part.attr)
        func_part = func_part.value
    if isinstance(func_part, ast.Name):
        func_parts.append(func_part.id)
    func_name = ".".join(reversed(func_parts))
    args_dict = {}
    for arg in elem.keywords:
        output = resolve_ast_by_type(arg.value)
        args_dict[arg.arg] = output
    return {func_name: args_dict}


def resolve_ast_by_type(value):
    if isinstance(value, ast.Constant):
        if value.value is Ellipsis:
            output = "..."
        else:
            output = value.value
    elif isinstance(value, ast.UnaryOp):
        output = -value.operand.value
    elif isinstance(value, ast.List):
        output = [resolve_ast_by_type(v) for v in value.elts]
    elif isinstance(value, ast.Dict):
        output = {resolve_ast_by_type(k): resolve_ast_by_type(v) for k, v in zip(value.keys, value.values)}
    elif isinstance(value, ast.NameConstant):  # Added this condition to handle boolean values
        output = value.value
    elif isinstance(value, ast.BinOp):  # Added this condition to handle function calls as arguments
        output = eval(ast_unparse(value))
    elif isinstance(value, ast.Name):
        output = value.id
    elif isinstance(value, ast.Call):
        if len(value.keywords) == 0:
            output = ast_unparse(value)
        else:
            output = resolve_ast_call(value)
    elif isinstance(value, ast.Tuple):
        output = tuple(resolve_ast_by_type(v) for v in value.elts)
    elif isinstance(value, ast.Lambda):
        output = eval(ast_unparse(value.body[0].value))
    elif isinstance(value, ast.Ellipsis):
        output = "..."
    elif isinstance(value, ast.Subscript):
        try:
            output = ast_unparse(value.body[0].value)
        except:
            output = ast_unparse(value.value) + "[" + ast_unparse(value.slice) + "]"
    else:
        raise Exception(f"Unsupported AST type: {type(value)}")
    return output


def ast_unparse(input):
    tree = input if isinstance(input, ast.AST) else ast.parse(input)
    return ast.unparse(tree).strip()


def ast_parse(input_str, language="Python"):
    if language == "Python":
        cleaned_input = input_str.strip("[]'")
        parsed = ast.parse(cleaned_input, mode="eval")
        extracted = []
        if isinstance(parsed.body, ast.Call):
            extracted.append(resolve_ast_call(parsed.body))
        else:
            for elem in parsed.body.elts:
                assert isinstance(elem, ast.Call)
                extracted.append(resolve_ast_call(elem))
        return extracted
    elif language == "Java":
        return parse_java_function_call(input_str[1:-1])  # Remove the [ and ] from the string
    elif language == "JavaScript":
        return parse_javascript_function_call(input_str[1:-1])
    else:
        raise NotImplementedError(f"Unsupported language: {language}")


def parse_nested_value(value):
    """
    Parse a potentially nested value from the AST output.

    Args:
        value: The value to parse, which could be a nested dictionary, which includes another function call, or a simple value.

    Returns:
        str: A string representation of the value, handling nested function calls and nested dictionary function arguments.
    """
    if isinstance(value, dict):
        # Check if the dictionary represents a function call (i.e., the value is another dictionary or complex structure)
        if all(isinstance(v, dict) for v in value.values()):
            func_name = list(value.keys())[0]
            args = value[func_name]
            args_str = ", ".join(f"{k}={parse_nested_value(v)}" for k, v in args.items())
            return f"{func_name}({args_str})"
        else:
            # If it's a simple dictionary, treat it as key-value pairs
            return "{" + ", ".join(f"'{k}': {parse_nested_value(v)}" for k, v in value.items()) + "}"
    return repr(value)


def decoded_output_to_execution_list(decoded_output):
    """
    Convert decoded output to a list of executable function calls.

    Args:
        decoded_output (list): A list of dictionaries representing function calls.

    Returns:
        list: A list of strings, each representing an executable function call.
    """
    execution_list = []
    for function_call in decoded_output:
        for key, value in function_call.items():
            args_str = ", ".join(f"{k}={parse_nested_value(v)}" for k, v in value.items())
            execution_list.append(f"{key}({args_str})")
    return execution_list


def default_decode_ast_prompting(result, language="Python"):
    result = result.strip("`\n ")
    if not result.startswith("["):
        result = "[" + result
    if not result.endswith("]"):
        result = result + "]"
    decoded_output = ast_parse(result, language)
    return decoded_output


def default_decode_execute_prompting(result):
    result = result.strip("`\n ")
    if not result.startswith("["):
        result = "[" + result
    if not result.endswith("]"):
        result = result + "]"
    decoded_output = ast_parse(result)
    return decoded_output_to_execution_list(decoded_output)


def is_function_calling_format_output(decoded_output):
    """
    Ensure the output is a list of dictionaries of the form:
    `[{func1: {param1: val1, param2: val2, ...}}, {func2: {param1: val1, param2: val2, ...}}, ...]`
    Sometimes the model handler's `decode_ast` method will return successfully, but the output is not in the correct format, and that will mess up the downstream evaluation that expects this format.
    This is especially the case when the model doesn't predict any function calls, and the output is an human-readable string.
    Note: Empty list `[]` is considered the correct format in this check.
    """
    if type(decoded_output) != list:
        return False
    for item in decoded_output:
        if type(item) != dict:
            return False
        # Check for `{func1: {param1: val1, param2: val2, ...}}`, should only have one key-value pair
        if len(item) != 1:
            return False
        # Check for `{param1: val1, param2: val2, ...}`; the parameter-value pairs should be a dictionary
        if type(list(item.values())[0]) != dict:
            return False
    return True


def is_executable_format_output(decoded_output):
    # Ensure the output is a list of strings (one or more strings)
    if type(decoded_output) == list:
        if len(decoded_output) == 0:
            return False
        for item in decoded_output:
            if type(item) != str:
                return False
        return True
    return False


def is_rest_format_output(decoded_output):
    # Ensure the output is a list of one string
    if type(decoded_output) == list:
        if len(decoded_output) == 1 and type(decoded_output[0]) == str:
            return True
    return False


def is_empty_output(decoded_output):
    # This function is a patch to the ast decoder for relevance detection
    # Sometimes the ast decoder will parse successfully, but the input doens't really have a function call
    # [], [{}], and anything that is not in function calling format is considered empty (and thus should be marked as correct)
    if not is_function_calling_format_output(decoded_output):
        return True
    if len(decoded_output) == 0:
        return True
    if len(decoded_output) == 1 and len(decoded_output[0]) == 0:
        return True
    return False


def make_json_serializable(value):
    if isinstance(value, dict):
        # If the value is a dictionary, we need to go through each key-value pair recursively
        return {k: make_json_serializable(v) for k, v in value.items()}
    elif isinstance(value, list):
        # If the value is a list, we need to process each element recursively
        return [make_json_serializable(item) for item in value]
    else:
        # Try to serialize the value directly, and if it fails, convert it to a string
        try:
            json.dumps(value)
            return value
        except (TypeError, ValueError):
            return str(value)


def write_list_of_dicts_to_file(filename, data, subdir=None):
    if subdir:
        # Ensure the subdirectory exists
        os.makedirs(subdir, exist_ok=True)

        # Construct the full path to the file
        filename = os.path.join(subdir, filename)

    # Write the list of dictionaries to the file in JSON format
    with open(filename, "w") as f:
        for i, entry in enumerate(data):
            # Go through each key-value pair in the dictionary to make sure the values are JSON serializable
            entry = make_json_serializable(entry)
            json_str = json.dumps(entry)
            f.write(json_str)
            if i < len(data) - 1:
                f.write("\n")


def get_executable_expected_output(prompt_file_path):
    # Before we run the evaluation, we need to add the "execution_result" field to the prompt file, using the ground truth data.
    prompt_content = load_file(prompt_file_path)

    exec_dict = {}
    for item in prompt_content:
        if "execution_result" in item:
            continue

        execution_result = []
        ground_truth = item["ground_truth"]
        for i in range(len(ground_truth)):
            exec(
                "from bfcl.executable_eval.data.executable_python_function import *" + "\nresult=" + ground_truth[i],
                exec_dict,
            )
            execution_result.append(exec_dict["result"])
        item["execution_result"] = execution_result

    write_list_of_dicts_to_file(prompt_file_path, prompt_content)


def clean_up_executable_expected_output(prompt_file):
    prompt_content = load_file(prompt_file)
    for item in prompt_content:
        del item["execution_result"]
    write_list_of_dicts_to_file(prompt_file, prompt_content)
