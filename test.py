import re

source = "lambda x: ''.join(re.findall('Final Revision:(.*)',x))"
# Convert lambda string to a real function without eval()

# Prepare a unique name for the function
func_str = f"func = {source}"

namespace = {'re': re}
exec(func_str, namespace)
my_func = namespace['func']

# Example usage:
x = "Some text Final Revision:2024-01 more text"
print(my_func(x))  # Output: 2024-01