# MatterGen Phase 2 Optimization Commands
# Add these aliases to your ~/.bashrc or ~/.zshrc

# Navigate to MatterGen directory
alias cdmattergen='cd /home/guillaume/Documents/Projects/LINK/Internships/Auguste/mattergenbis'

# Direct command aliases
alias mattergen-generate='cd /home/guillaume/Documents/Projects/LINK/Internships/Auguste/mattergenbis && python -c "import sys; sys.path.insert(0, \".\"); from mattergen.scripts.generate import main; main()"'

# Quick optimization commands
alias mattergen-optimized='cd /home/guillaume/Documents/Projects/LINK/Internships/Auguste/mattergenbis && python -c "import sys; sys.path.insert(0, \".\"); from mattergen.scripts.generate import main; main()" results/ --pretrained-name=mattergen_base --batch_size=64 --sampling_config_name=optimized --enable_optimizations=True'

alias mattergen-fast='cd /home/guillaume/Documents/Projects/LINK/Internships/Auguste/mattergenbis && python -c "import sys; sys.path.insert(0, \".\"); from mattergen.scripts.generate import main; main()" results/ --pretrained-name=mattergen_base --batch_size=128 --sampling_config_name=fast --enable_optimizations=True'

# Usage:
# source this file: source mattergen_aliases.sh
# Then reload your shell or run: source ~/.bashrc
