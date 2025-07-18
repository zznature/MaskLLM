# Project Cursor Rules

## Command Execution Protocol

### Problem Statement
- Server command execution is slow
- Interactive command-by-command approach is inefficient
- Need a more streamlined workflow

### Solution Protocol
1. **Command Batching**
   - Assistant will compile all necessary commands into a single script file
   - Commands will be properly sequenced with appropriate error handling
   - Each script will be self-contained with clear purpose

2. **Execution Process**
   - User will execute the compiled script file
   - User will share execution results
   - Assistant will analyze results and plan next steps

3. **Script Requirements**
   - Include clear comments explaining each command's purpose
   - Add appropriate error checking where possible
   - Use consistent naming convention: `task_name_commands.sh`
   - Begin each script with proper shebang (`#!/bin/bash`)
   - Set appropriate permissions (`chmod +x`)

4. **Result Analysis**
   - Assistant will thoroughly analyze script execution results
   - Identify any errors or unexpected outcomes
   - Propose next steps based on results
   - Create new command scripts as needed

5. **Documentation**
   - Keep track of successful commands and their outcomes
   - Document any persistent issues or workarounds
   - Update project status after each successful step

This protocol will minimize server interaction overhead while maintaining effective troubleshooting and development workflow. 