# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from pathlib import Path


TRAINING_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "training"
if str(TRAINING_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_SCRIPT_DIR))

from run_recipe import main as run_recipe_main  # noqa: E402


def main() -> None:
    """Run the unified recipe entry point in benchmark compatibility mode."""
    run_recipe_main(benchmark=True)


if __name__ == "__main__":
    main()
