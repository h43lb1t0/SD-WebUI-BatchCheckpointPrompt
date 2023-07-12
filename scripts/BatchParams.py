from dataclasses import dataclass
from typing import Union, List, Tuple
import re
import os

from scripts.Logger import Logger
from scripts.Utils import Utils

import itertools

import modules
import modules.shared as shared

@dataclass()
class BatchParams:
    checkpoint: str
    prompt: str
    neg_prompt: str
    batch_count: int = -1
    clip_skip: int = 1

    def __repr__(self):
        checkpointName: str = os.path.basename(self.checkpoint)
        return( f"BatchParams: {checkpointName},\n " 
               f"prompt: {self.prompt},\n"
               f"neg_prompt: {self.neg_prompt},\n "
               f"batch_count: {self.batch_count},\n "
               f"clip_skip: {self.clip_skip}\n")


def get_all_batch_params(p: Union[modules.processing.StableDiffusionProcessingTxt2Img, modules.processing.StableDiffusionProcessingImg2Img], checkpoints_as_string: str, prompts_as_string: str, cycle_prompts: bool, enable_iterators: bool) -> List[BatchParams]:

        logger = Logger()
        logger.debug = False
        utils = Utils()

        def get_batch_count_from_prompt(prompt: str) -> Tuple[int, str]:
            # extracts the batch count from the prompt if specified
            number_match = re.search(r"\{\{count:([0-9]+)\}\}", prompt)
            if number_match and number_match.group(1):
                # Extract the number from the match object
                number = int(number_match.group(1))  # Use group(1) to get the number inside parentheses
                number = p.n_iter if number < 1 else number
                prompt = re.sub(r"\{\{count:[0-9]+\}\}", '', prompt)
            else:
                number = p.n_iter


            return number, prompt
        
        def get_clip_skip_from_prompt(prompt: str) -> Tuple[int, str]:
            # extracts the clip skip from the prompt if specified
            number_match = re.search(r"\{\{clip_skip:([0-9]+)\}\}", prompt)
            if number_match and number_match.group(1):
                # Extract the number from the match object
                number = int(number_match.group(1))
                number = shared.opts.data["CLIP_stop_at_last_layers"] if number < 1 else number
                prompt = (
                    re.sub(r"\{\{clip_skip:[0-9]+\}\}", '', prompt))
            else:
                number = shared.opts.data["CLIP_stop_at_last_layers"]

            return number, prompt

        def split_postive_and_negative_postive_prompt(prompt: str) -> Tuple[str, str]:
            pattern = r'{{neg:(.*?)}}'
            # Split the input string into parts
            parts = re.split(pattern, prompt)
            if len(parts) > 1:
                neg_prompt = parts[1]
            else:
                neg_prompt = ""

            prompt = parts[0]

            return prompt, neg_prompt


        all_batch_params: List[BatchParams] = []

        if cycle_prompts or enable_iterators:
            checkpoints_as_string = apply_checkpoint_iteration_count_from_prompt(checkpoints_as_string)

        checkpoints: List[str] = utils.getCheckpointListFromInput(checkpoints_as_string)


        prompts: List[str] = utils.remove_index_from_string(prompts_as_string).split(";")
        prompts = [prompt.replace('\n', '').strip() for prompt in prompts if not prompt.isspace() and prompt != '']

        #if this is enabled, the prompt list will be made to match the checkpoint list in length by looping or truncating
        if cycle_prompts:
            # Prepare an infinite iterator from the prompts list
            infinite_prompts_iterator: Iterator = itertools.cycle(prompts)
            # convert the infinite iterator to a checkpoint-length iterator
            checkpoint_length_prompts_iterator: Iterator = itertools.islice(infinite_prompts_iterator, len(checkpoints))
            # convert the checkpoint-length iterator to a list of prompts that matches the list of checkpoints in length 
            prompts: List[str] = list(checkpoint_length_prompts_iterator)
        else:
            if len(prompts) != len(checkpoints):
                logger.debug_log(f"len prompt: {len(prompts)}, len checkpoints{len(checkpoints)}")
                raise RuntimeError(f"amount of prompts don't match with amount of checkpoints")


        if len(prompts) == 0:
            raise RuntimeError(f"can't run without a checkpoint and prompt")
        
        # if enabled, this replaces syntax for variables that change from generation to generation
        # i.e. [0, varName, +=, 10] translates into 0 as the first value, then 10, then 20, etc.
        if enable_iterators:
            prompts = process_variables_from_prompts(prompts)

        
        for i in range(len(checkpoints)):

            info = modules.sd_models.get_closet_checkpoint_match(checkpoints[i])
            if info is None:
                raise RuntimeError(f"Unknown checkpoint: {checkpoints[i]}")


            batch_count, prompts[i] = get_batch_count_from_prompt(prompts[i])
            clip_skip, prompts[i] = get_clip_skip_from_prompt(prompts[i])
            prompt, neg_prompt = split_postive_and_negative_postive_prompt(prompts[i])


            prompt = prompt.replace("{prompt}", p.prompt)
            neg_prompt = neg_prompt = p.negative_prompt + neg_prompt


            all_batch_params.append(BatchParams(checkpoints[i], prompt, neg_prompt, batch_count, clip_skip))

        return all_batch_params





############################# Code To Process Prompt Operands/Iterators #############################




# I could have been less verbose here, but
#     I didn't want to push the case sensitivity logic
#     into another function

# TO ADD TO THIS LIST, ADD AN ENTRY TO THE OPERATORS DICTIONARY
#   EXTERNAL FUNCTIONS ARE JUST NEEDED IF YOU WANT TO COVER DIFFERENT CASES OR SYNONYMS
#   YOU CANNOT INCLUDE COMMAS IN THE OPERATOR STRINGS
#   CURRENTLY LAMBDAS ARE LIMITED TO ONE VARIABLE BY THE CALLING CODE
#      THAT DOESN'T MEAN YOU CAN'T ADD THEM HERE, BUT YOU'LL NEED TO EXPAND THE CALLING CODE

# Define the operations

sine = lambda x, amplitude=1, frequency=1 : amplitude * math.sin(frequency * x)
cosine = lambda x, amplitude=1, frequency=1 : amplitude * math.cos(frequency * x)
tangent = lambda x, amplitude=1, frequency=1 : amplitude * math.tan(frequency * x)
arcsine = lambda x, amplitude=1, frequency=1 : amplitude * math.asin(frequency * x)
arccosine = lambda x, amplitude=1, frequency=1 : amplitude * math.acos(frequency * x)
arctangent = lambda x, amplitude=1, frequency=1 : amplitude * math.atan(frequency * x)
square_root = lambda x: math.sqrt(x)
log = lambda x, amplitude=1, frequency=1 : amplitude * math.log(frequency * x)
log10 = lambda x, amplitude=1, frequency=1 : amplitude * math.log10(frequency * x)
log2 = lambda x, amplitude=1, frequency=1 : amplitude * math.log2(frequency * x)
approach_limit = lambda x, limit=1, rate=1: x + (limit - x) * 0.5 * rate

Operators = {
   # ADD NEW OPERATORS HERE
   #'operator': operator_function,
    'default': lambda x: x,
    '++': lambda x: x + 1,
    '--': lambda x: x - 1,
    '+=': lambda x, n: x + n,
    '-=': lambda x, n: x - n,
    '*=': lambda x, n: x * n,
    '/=': lambda x, n: x / n,
    '%=': lambda x, n: x % n,
    '^=': lambda x, n: x ** n,
    'sin': sine,
    'Sin': sine,
    'SIN': sine,
    'cos': cosine,
    'Cos': cosine,
    'COS': cosine,
    'tan': tangent,
    'Tan': tangent,
    'TAN': tangent,
    'asin': arcsine,
    'Asin': arcsine,
    'ASIN': arcsine,
    'acos': arccosine,
    'Acos': arccosine,
    'ACOS': arccosine,
    'atan': arctangent,
    'Atan': arctangent,
    'ATAN': arctangent,
    'sqrt': square_root,
    'Sqrt': square_root,
    'SQRT': square_root,
    'log': log,
    'Log': log,
    'LOG': log,
    'log10': log10,
    'Log10': log10,
    'LOG10': log10,
    'log2': log2,
    'Log2': log2,
    'LOG2': log2,
    'approach_limit': approach_limit,
    'Approach_Limit': approach_limit,
    'APPROACH_LIMIT': approach_limit
}





import math
import inspect

# Define the pattern for the operation string 
#     [initial value,promptIteratorName,function,optional n value, optional a value, optional b value, optional c value, optional d value, optional...]
#     example: [0,varName,+=,10] translates into 0 as the first value, then 10, then 20, etc.
#     see OperandFunctionDefinitions.py for the list of included functions

def round_to_significant_digits(num, sig_figs):
    try:
        num = float(num)
        if num != 0:
            return round(num, sig_figs - int(math.floor(math.log10(abs(num)))) - 1)
        else:
            return 0  # Can't take the log of 0
    except ValueError:
        return num

def format_number_from_string(number, reference_string):
    try:
        number_float = float(number)
    except ValueError:
        print(f"number: {number} is not a float or int or cannot be cast to a float, returning str(number)")
        return str(number)    

    if not isinstance(reference_string, str):
        print(f"reference_string: {reference_string} is not a string, returning str(number) {str(number)}")
        return str(number)    
    try:
        float(reference_string)
    except ValueError:
        print(f"reference_string: {reference_string} cannot parse to a float, returning str(number) {str(number)}")
        return str(number)   
   
    reference_integer, reference_dec_point, reference_decimal = '','',''
    number_integer, number_dec_point, number_decimal = '','',''
    
    try:
        if '.' in str(reference_string):
            reference_integer, reference_decimal = reference_string.split('.')
            reference_dec_point = '.'            
        else:
            reference_integer = reference_string
            reference_dec_point = ''
    except ValueError:
        return str(number)
    
    try:
        if '.' in str(number).rstrip('0').rstrip('.'):
            number_integer, number_decimal = (str(number).rstrip('0').rstrip('.')).split('.')
            number_dec_point = '.'            
        else:
            number_integer, number_decimal = str(number), ''
            number_dec_point = ''
    except ValueError:
        return str(number)
    number_integer = number_integer.rstrip('0').rstrip('.')
    number_decimal.rstrip('0').rstrip('.').lstrip('0').lstrip('.')
    while len(number_decimal) < len(reference_decimal):
        number_decimal = number_decimal + '0'
    
    while len(number_integer) < len(reference_integer):
        number_integer = '0' + number_integer
    
    decimal_point = ''
    if number_dec_point == '.' or reference_dec_point == '.':
        decimal_point = '.'
    
    formatted_number = number_integer + decimal_point + number_decimal
    
    return formatted_number
    
RGX_CHECKPOINT_ITERATIONS = '\[checkpointIterations=([0-9]+)\]'


def apply_checkpoint_iteration_count_from_prompt(original_checkpoint_string):
    checkpoint_string = original_checkpoint_string
    if checkpoint_string is None or checkpoint_string == '':
        return original_checkpoint_string 

    #find each occurrence of RGX_CHECKPOINT_ITERATIONS in checkpoint_string
    checkpoint_iteration_matches = re.findall(RGX_CHECKPOINT_ITERATIONS, checkpoint_string)
    if checkpoint_iteration_matches is None or checkpoint_iteration_matches == []:
        return original_checkpoint_string
    
    #get checkpoint_iteration_count from the first match
    checkpoint_iteration_count_str = checkpoint_iteration_matches[0]
    checkpoint_iteration_count = 1
    try:
        checkpoint_iteration_count = int(checkpoint_iteration_count_str)
    except ValueError:
        print(f"checkpoint_iteration_count_str: {checkpoint_iteration_count_str} is not an int, returning checkpoint_string: {checkpoint_string}")
        
    #remove any leading or trailing commas (that will prevent the split from generating an error)
    checkpoint_string.strip(',')

    try:
        split_checkpoints = checkpoint_string.split(',')
    except ValueError:
        print(f"checkpoint_string: {checkpoint_string} could not be split by commas, returning checkpoint_string: {checkpoint_string}")
        return original_checkpoint_string
    
    if checkpoint_iteration_count <1:
        print(f"checkpoint_iteration_count: {checkpoint_iteration_count} is less than 1, no effect is applied: {checkpoint_string}")
        
    new_checkpoints = []
    #for each checkpoint in checkpoints, add it to new_checkpoints checkpoint_iteration_count times
    for checkpoint in split_checkpoints:
        for i in range(checkpoint_iteration_count):
            #do not add RGX_CHECKPOINT_ITERATIONS matches to the new list
            RGX_matches = re.findall(RGX_CHECKPOINT_ITERATIONS, checkpoint)
            if RGX_matches is None or RGX_matches == []:
                new_checkpoints.append(checkpoint)            
    
    checkpoint_string = ','.join(new_checkpoints)

    return checkpoint_string
    


RGX_ITERATOR_PATTERN = '\[[^,\[\]]+(?:,[^,\[\]]+)*\]'

def parse_operation_str(operation_str):
    def is_valid_string(variable):
        pattern = r'^[a-zA-Z0-9_-]+$'
        return bool(re.match(pattern, variable))

    def is_valid_operand_characters(variable):
        pattern = r'[^,\s]+$'
        return bool(re.match(pattern, variable))
    
    operation_match = re.match(RGX_ITERATOR_PATTERN, operation_str)

    if operation_match is None:
        return operation_str
    
    #remove the brackets
    operation_match = operation_match.group()[1:-1]

    #remove any trailing whitespace or commas
    while operation_match and (operation_match[-1] == ',' or operation_match[-1] == ' '):
        operation_match = operation_match[:-1]


    #split and strip the string into its components
    operation_arguments = operation_match.split(',')
    operation_arguments = [arg.strip() for arg in operation_arguments]
    
    initial_value_format_str = '0'
    try:
        initial_value_format_str = operation_arguments[0]
        print(f"initial_value_format_str: {initial_value_format_str}")     
        initial_value_str_to_cast_to_float = initial_value_format_str
        initial_value = float(initial_value_str_to_cast_to_float)
    except ValueError:
        return operation_str
    
    variable_name = operation_arguments[1]
    if is_valid_string(variable_name) is False:
        return operation_str
    
    operator = operation_arguments[2]
    if is_valid_operand_characters(operator) is False:
        return operation_str
    
    operands = []
    #assign remaining arguments to operands, escape at the first invalid operand
    for operand in operation_arguments[3:]:
        try:
            floatOperand = float(operand)
            operands.append(floatOperand)
        except ValueError:
            break

    #if operands is of type collection, but is empty, set it to None
    if isinstance(operands, list) and len(operands) == 0:
        operands = None

    return initial_value, variable_name, operator, operands, initial_value_format_str
    

def run_lambda_despite_too_many_args(lambda_function, args):
    passed_args = args
    
    lambda_function_arg_count = len(inspect.getfullargspec(lambda_function).args)
    #trim the arg length
    passed_args = passed_args[:lambda_function_arg_count]
    
    #while last passed ark is None, trim it
    while passed_args and passed_args[-1] is None:
        passed_args = passed_args[:-1]

    #run round_to_significant_digits for each item in passed args
    passed_args = [round_to_significant_digits(arg, 5) for arg in passed_args]

    try:
        lambda_result = lambda_function(*passed_args)
        return round_to_significant_digits(lambda_result, 12)
    except TypeError:
        print(f"Lambda or rounding function encountered a TypeError. Lambda result: {lambda_result}, args: {args}")
        return args[0]
    
def process_variables_from_prompts(prompts):
    #dictionary format is {variable_name: (initial/iterating value, operator, operands)}
    variables_dict = {}
    processed_prompts = []
    for prompt in prompts:        
        matches = re.findall(RGX_ITERATOR_PATTERN, prompt)
        for match in matches:
            operation_str = match

            parsed_operation_args = parse_operation_str(operation_str)

            # If parsed_operation is a string, do nothing
            if isinstance(parsed_operation_args, str):
                print(f"parse_operation_str returned a string, leaving unchanged: {parsed_operation_args}")                
                continue
           
            parsed_iterator_value, parsed_var_name, parsed_operator, parsed_operands, initial_value_format_str = parsed_operation_args
            print(f"parsed from custom iterator: {parsed_iterator_value} {parsed_var_name} {parsed_operator} {str(parsed_operands)}")
            if parsed_var_name in variables_dict:
                #parsed operator and operands can change, but initial value is ignored after first detection
                if parsed_operator in Operators:
                    operation_lambda = Operators[parsed_operator]
                else:
                    print(f"[{parsed_operator} not found in operators, returning initial value: {parsed_iterator_value}")
                    prompt = prompt.replace(match,str(parsed_iterator_value).rstrip('0').rstrip('.'), 1)
                    continue
                stored_iterator_value = variables_dict[parsed_var_name][0]
                try:
                    stored_iterator_value = float(stored_iterator_value)
                except ValueError:
                    print(f"stored_iterator_value: {stored_iterator_value} is not a float, returning initial value: {parsed_iterator_value}")
                    prompt = prompt.replace(match,str(parsed_iterator_value).rstrip('0').rstrip('.'), 1)
                    continue
                lambda_args = [stored_iterator_value, *(parsed_operands if parsed_operands is not None else [])]

                parsed_iterator_value = run_lambda_despite_too_many_args(operation_lambda, lambda_args)
            
            #remove any trailing zeros or decimals
            parsed_iterator_value = str(parsed_iterator_value).rstrip('0').rstrip('.')
            #if variable is not in the dictionary, add it but it doesn't get processed the first time it's encountered
            variables_dict[parsed_var_name] = [parsed_iterator_value, parsed_operator, parsed_operands]
            
            formatted_iterator_value_str = format_number_from_string(variables_dict[parsed_var_name][0], initial_value_format_str)
            
            prompt = prompt.replace(match,formatted_iterator_value_str, 1)
            print(f"New prompt after iterator applied: {prompt}")
            
        processed_prompts.append(prompt)

    return processed_prompts