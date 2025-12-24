import re
text = "My phone number is 987-654-3210."
match = re.search(r"\d{3}-\d{3}-\d{4}", text) # \d{3} meaning exactly 3 digits then one -(hyphen) and again \d{3} - \d{4}
print(match.group())  # Output: 987-654-3210

print('''Pattern	Description	Example
\d	Matches any digit (0–9)	There are 3 cats.
\w	Matches any word character (letters, digits, _)	var_1 = 10
\s	Matches any whitespace character	New\nLine
^hello	Matches 'hello' at the start of a string	hello world
world$	Matches 'world' at the end of a string	It's a small world
\bword\b	Matches the whole word 'word'	The word is powerful.
\d{3}-\d{3}-\d{4}	Matches phone number like 123-456-7890	My phone number is 123-456-7890
\(\d{3}\)-\d{3}-\d{4} Matches phone number like (123)-456-7890.
[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}	Matches email addresses	Contact us at support@mail.com
\b[A-Z][a-z]*\b	Matches words that start with a capital letter	Hello from India.
[^\w\s]	Matches special characters (not letter/number)	Hello@2023!
''')

text = "You can contact me at (123)-456-7890 for more information."
pattern = r"\(\d{3}\)-\d{3}-\d{4}"

match = re.search(pattern, text)
if match:
    print("Phone number found:", match.group())
else:
    print("No phone number found.")


text = """
Contact John at (123)-456-7890.
For support, call (987)-654-3210 or email help@example.coming.
You can also reach (555)-123-4567 for urgent queries.
"""

patternS = "\(\d{3}\)-\d{3}-\d{3}"
email_patterns = "[a-zA-Z]*@[A-Za-z]*.[A-Za-z]{3}" # * represent one or more alpha characters

matches = re.findall(patternS,text)
emails  = re.findall(email_patterns,text)

print('Below are the phone numbers found in the text:')
for phone in matches:
    print(phone)

print('Emails found in the text are:')
for mail in emails:
      print(mail)

text = "Hello@2023! Welcome to NLP, AI & ML: Let's learn fast."
remove_spcial = "[a-zA-Z]\S+|\@"

clean_text = re.findall(remove_spcial,text)
print('Cleaned text now is:',clean_text)

#Python Code to Remove Special Characters:
import re

text = "Hello@2023! Welcome to NLP, AI & ML: Let's learn fast."

# Keep only alphabetic characters and spaces
clean_text = re.sub(r'[^a-zA-Z\s]', '', text)

print("Cleaned text:", clean_text)

text = "My email is user123@example.com and website is https://www.openai.com.This one is another email-> john.doe123@example.co.in , this is \
my firend email addres vissenthil@gmail.com"

comm_email_pattern =r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
URL_pattern = r'https?:\/\/(www\.)?[a-zA-Z0-9\-]+\.[a-zA-Z]{2,}(\/\S*)?'
emails = re.findall(comm_email_pattern,text)
urls = re.findall(URL_pattern, text)

print(emails)
print('URL:',urls)

'''
Matches emails like: john.doe123@example.co.in
Step-by-Step Explanation
1. \b
Meaning: Word boundary

Why: Ensures that the email starts at the boundary of a word (e.g., doesn't match abc@xyz.com inside 123abc@xyz.com123)

2. [A-Za-z0-9._%+-]+
Meaning: Username part of the email

Matches: One or more characters from the set:
A-Z, a-z → letters
0-9 → digits
._%+- → common email username characters

🔍 Examples matched:
john.doe, user_123, first+last, info%mail

3. @
Meaning: Literal @ symbol

Why: Separates the username from the domain

4. [A-Za-z0-9.-]+
Meaning: Domain name

Matches: One or more letters, digits, dots (.), or hyphens (-)

🔍 Examples matched:
gmail, yahoo, openai, my-company.co

5. \.
Meaning: Literal dot .

Why escaped?: Dot . is a special regex character (matches any character), so we escape it to match an actual period.

6. [A-Za-z]{2,}
Meaning: Top-level domain (TLD)

Matches: At least 2 or more letters

✅ Matches: .com, .org, .edu, .co.in, .tech

🔍 [A-Za-z]{2,} → means:

Must be at least 2 characters (like in, uk)

Only letters allowed

7. \b
Meaning: Ending word boundary

Why: Makes sure the email ends cleanly (e.g., doesn’t capture extra characters like john@mail.com123)


'''
from datetime import datetime
import pandas as pd
print('Date Regular expressions:')

text = """
Today is 16-05-2025. The previous update was on 12/04/2024.
A log entry on 05.03.2023 shows system activity. Invalid date: 32/13/2025.
Event dates: 01-01-2022, 15/08/1947, and 30.12.1999. I was born in  15-May-1975.
"""

# Regex pattern to match dd-mm-yyyy, dd/mm/yyyy, dd.mm.yyyy
date_pattern = r'\b\d{2}[-/.]\d{2}[-/.]\d{4}\b|\b\d{2}[-/.][A-Za-z]{3}[-/.]\d{4}\b' 
#it will extract two type of dates (30.12.1999,01-01-2022, 15/08/1947) or 15-May-1975.

# Find all matching dates
dates = re.findall(date_pattern, text)

def is_valid_format(date_str, format="%d-%b-%Y"):
    try:
        datetime.strptime(date_str, format)
        return True
    except ValueError:
        return False
'''
# above function to Test
print(is_valid_format("15-May-1975"))  # ✅ True
print(is_valid_format("15-05-1975"))   # ❌ False
print(is_valid_format("05/Jul/2023"))  # ❌ False

In Python’s datetime formatting (strftime and strptime), %b stands for:

✅ %b → Abbreviated month name
Examples: Jan, Feb, Mar, ..., Dec

'''
    
# Step 1: Parse the original date
str_date =''
parsed_date =[]
for date in dates:
   str_date = date
   if is_valid_format(str_date):
        date_ls = datetime.strptime(str_date, "%d-%b-%Y")
        parsed_date.append(date_ls.strftime("%d-%m-%Y"))
   else:     
         parsed_date.append(date)
# Step 2: Format into desired form

print(parsed_date)

# Print results
print("Dates found:")
for date in parsed_date:
    print(date)

