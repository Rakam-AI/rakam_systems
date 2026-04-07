# Changelog

All notable changes to this project will be documented in this file. See [conventional commits](https://www.conventionalcommits.org/) for commit guidelines.

---

## [0.3.1](https://github.com/Rakam-AI/rakam_systems/compare/0.3.0..0.3.1) - 2026-04-02

### Features

- **(agent)** restore **init** and update exported classes/modules - ([7cd4817](https://github.com/Rakam-AI/rakam_systems/commit/7cd4817b99bb4646f36dbd75a912f5728c819071)) - [somebodyawesome-dev](https://github.com/somebodyawesome-dev)
- **(ci)** added display coverage step into ci - ([0559a7f](https://github.com/Rakam-AI/rakam_systems/commit/0559a7f3c8272c2876817c7179009e1d57e6e4ad)) - [somebodyawesome-dev](https://github.com/somebodyawesome-dev)
- **(ci)** update coverage step - ([1f6164b](https://github.com/Rakam-AI/rakam_systems/commit/1f6164be3416b0075bed0adea87f7409e6adefe0)) - [somebodyawesome-dev](https://github.com/somebodyawesome-dev)
- **(release)** enhance release workflow with package-specific toggles and version bumping - ([e8fd9e3](https://github.com/Rakam-AI/rakam_systems/commit/e8fd9e3b507d06ded9bc9161369baa3067f874d9)) - [somebodyawesome-dev](https://github.com/somebodyawesome-dev)
- **(release)** streamline tagging process for rakam-systems - ([467c61b](https://github.com/Rakam-AI/rakam_systems/commit/467c61bdfcf9335c48b34f8af12dbacfb0730d7e)) - [somebodyawesome-dev](https://github.com/somebodyawesome-dev)
- **(release)** simplify tag naming for rakam-systems - ([d35d1ce](https://github.com/Rakam-AI/rakam_systems/commit/d35d1ce407134380acc25fb980559ac80c0aedea)) - [somebodyawesome-dev](https://github.com/somebodyawesome-dev)
- add release workflow for versioning and publishing - ([49808ab](https://github.com/Rakam-AI/rakam_systems/commit/49808ab3d49f4ff91010b8a6bc5342307a82b23a)) - [somebodyawesome-dev](https://github.com/somebodyawesome-dev)
- add message_history parameter to async inference methods in BaseAgent - ([94d38e1](https://github.com/Rakam-AI/rakam_systems/commit/94d38e18062c90ffc8f90661898becc446ec3587)) - [somebodyawesome-dev](https://github.com/somebodyawesome-dev)
- update build system to use uv_build and execluded tests from builds - ([17c4684](https://github.com/Rakam-AI/rakam_systems/commit/17c4684fa1ad211436a7c656d36858d6ffec306a)) - [somebodyawesome-dev](https://github.com/somebodyawesome-dev)
- **(rakam_systems)** initialize **init**.py with module imports and aliases - ([19c79ef](https://github.com/Rakam-AI/rakam_systems/commit/19c79efb2be7add3b5393cdad5d393835c61d77d)) - [somebodyawesome-dev](https://github.com/somebodyawesome-dev)

### Bug Fixes

- **(docs)** fix broken links in docs - ([e148267](https://github.com/Rakam-AI/rakam_systems/commit/e14826703b48dc05f52b139e6d42e07dbbddfcb0)) - [somebodyawesome-dev](https://github.com/somebodyawesome-dev)
- **(docs)** accurate dependencies, install info, and DG getting-started - ([335d9e5](https://github.com/Rakam-AI/rakam_systems/commit/335d9e5d2c400a0f39dba53441dcbe33ba94b801)) - [Yann Rapaport](https://github.com/YannRapaport)
- **(evaluation)** Correct spelling of 'expected_schema' in JsonCorrectnessConfig and FieldsPresenceConfig - ([3273e1e](https://github.com/Rakam-AI/rakam_systems/commit/3273e1e5220a404cb24ac8083823f54d1c797d19)) - [somebodyawesome-dev](https://github.com/somebodyawesome-dev)
- **(tests)** fix coverage depencies for ci - ([9ac05c7](https://github.com/Rakam-AI/rakam_systems/commit/9ac05c728ac6ba69e5e25cea95be93a8e3f3d8a8)) - [somebodyawesome-dev](https://github.com/somebodyawesome-dev)
- **(vectorstore)** correct typo in warning message for empty nodes in FAISS index - ([594ae3b](https://github.com/Rakam-AI/rakam_systems/commit/594ae3b4c3d18c4620593d9c0114c0853ee0d5c4)) - [somebodyawesome-dev](https://github.com/somebodyawesome-dev)
- **(vectorstore)** Fixed unclosed '{' in one of the unittests - ([772e229](https://github.com/Rakam-AI/rakam_systems/commit/772e229d0d0b7b31bd161870c1b5c85c73033b5b)) - [somebodyawesome-dev](https://github.com/somebodyawesome-dev)

### Documentation

- **(agent)** update agents section to include chat history functionality - ([8ce40f1](https://github.com/Rakam-AI/rakam_systems/commit/8ce40f1a9c7a17cb742eb36645a8374ab7b45dfa)) - [somebodyawesome-dev](https://github.com/somebodyawesome-dev)
- **(changelog)** added cliff config and generated changelog file - ([5f81b3f](https://github.com/Rakam-AI/rakam_systems/commit/5f81b3f54055362d95d42d0686e3cd9b589ab29f)) - [somebodyawesome-dev](https://github.com/somebodyawesome-dev)
- **(changelog)** update cliff template - ([b69b062](https://github.com/Rakam-AI/rakam_systems/commit/b69b062e17cbc5dc284ff4470d5c202a22ae723c)) - [somebodyawesome-dev](https://github.com/somebodyawesome-dev)
- mark observability features as experimental - ([816dfe5](https://github.com/Rakam-AI/rakam_systems/commit/816dfe5246bf5020ee49a5be39718e8cef2c3d3a)) - [Yann Rapaport](https://github.com/YannRapaport)
- draft Contributing section (code of conduct, development, documentation) - ([2603510](https://github.com/Rakam-AI/rakam_systems/commit/2603510175b92d04d0a0992144f266f701128881)) - [Yann Rapaport](https://github.com/YannRapaport)
- split User Guide and Developer Guide into multi-page categories - ([1882986](https://github.com/Rakam-AI/rakam_systems/commit/188298687eecdf15b97b4d08fd8f26bf7823fbb5)) - [Yann Rapaport](https://github.com/YannRapaport)
- extract Getting Started sub-pages from guide index pages - ([40e2543](https://github.com/Rakam-AI/rakam_systems/commit/40e25438ae9bed925ec75bfc619c116d40db6e29)) - [Yann Rapaport](https://github.com/YannRapaport)
- fix wrong imports and wrong content in READMEs - ([6f89694](https://github.com/Rakam-AI/rakam_systems/commit/6f8969428298b7e403890ba3a6ca3232e754dc12)) - [Yann Rapaport](https://github.com/YannRapaport)
- fix broken Docker paths and dead links in MCP READMEs - ([c54081a](https://github.com/Rakam-AI/rakam_systems/commit/c54081aa4cc8333d4c6bd05f6f359543643a3e20)) - [Yann Rapaport](https://github.com/YannRapaport)
- trim duplicated content and add official docs links - ([9a3c958](https://github.com/Rakam-AI/rakam_systems/commit/9a3c958f2fa0368e3d8a316edf876b1df5bfcffd)) - [Yann Rapaport](https://github.com/YannRapaport)
- \*_(tests)_- clean up getting started guide (QSG-MINOR-01, 05, 06, 07) - ([b8a3251](https://github.com/Rakam-AI/rakam_systems/commit/b8a32518dd023240c45ceb55fc51afcebfa78933)) - [Yann Rapaport](https://github.com/YannRapaport)
- rewrite User Guide — structure, accuracy, and content cleanup - ([f8c7909](https://github.com/Rakam-AI/rakam_systems/commit/f8c7909a0ef71483b698e9af3616a8cde3161ced)) - [Yann Rapaport](https://github.com/YannRapaport)
- remove Reference Guide (content merged into User Guide) - ([03832f1](https://github.com/Rakam-AI/rakam_systems/commit/03832f19a3e2bcfcec7ac93530b6284a126e4eb3)) - [Yann Rapaport](https://github.com/YannRapaport)
- consistency pass on Developer Guide, Getting Started, and Introduction - ([ad1cba0](https://github.com/Rakam-AI/rakam_systems/commit/ad1cba0f6380853d4900f945c55bfe8f4084c5a2)) - [Yann Rapaport](https://github.com/YannRapaport)
- merge Eval SDK and S3 into User Guide, expand CLI reference - ([31980e9](https://github.com/Rakam-AI/rakam_systems/commit/31980e9c2454594cda0c5c80d8f0a3d578550c98)) - [Yann Rapaport](https://github.com/YannRapaport)
- restructure Developer Guide as progressive teaching guide - ([d818dc9](https://github.com/Rakam-AI/rakam_systems/commit/d818dc9649f902173044a7fbb5e06c644b7227a2)) - [Yann Rapaport](https://github.com/YannRapaport)
- add PostgreSQL setup instructions to vectorstore docs - ([32b3cbc](https://github.com/Rakam-AI/rakam_systems/commit/32b3cbcc69bbc6d1b01e29db5503e2abd0806ba9)) - [Yann Rapaport](https://github.com/YannRapaport)
- restore experimental markings removed by docs v2 merge - ([e3b2585](https://github.com/Rakam-AI/rakam_systems/commit/e3b2585eaf36cb0207803d32316751a2010f7ef2)) - [Yann Rapaport](https://github.com/YannRapaport)\* enhance agent tests with mock_agent_run for consistent behavior - ([c731336](https://github.com/Rakam-AI/rakam_systems/commit/c731336225dfb1c5729c5f2d3a38e0600938fc72)) - [somebodyawesome-dev](https://github.com/somebodyawesome-dev)

### Miscellaneous Chores

- **(tools)** update version to 0.2.1 and initialize module exports - ([d3df11e](https://github.com/Rakam-AI/rakam_systems/commit/d3df11e2410ab17863b1bfd0910453e2b042f7ee)) - [somebodyawesome-dev](https://github.com/somebodyawesome-dev)
- **(agent)** update version to 0.1.4 and adjust dependencies in pyproject.toml and uv.lock - ([155491a](https://github.com/Rakam-AI/rakam_systems/commit/155491ae79ef12341495388d7bbc7e2a6d1db39c)) - [somebodyawesome-dev](https://github.com/somebodyawesome-dev)
- **(vectorstore)** update version to 0.1.3 and adjust dependency constraints in pyproject.toml and uv.lock - ([b244d5f](https://github.com/Rakam-AI/rakam_systems/commit/b244d5f36875bbc5d13af269463bb704b64da531)) - [somebodyawesome-dev](https://github.com/somebodyawesome-dev)
- **(cli)** update version to 0.2.6 and adjust dependencies in pyproject.toml and uv.lock - ([9b4b0a0](https://github.com/Rakam-AI/rakam_systems/commit/9b4b0a00ffabd69830d44cf135c5b5f90f0f71c0)) - [somebodyawesome-dev](https://github.com/somebodyawesome-dev)
- **(cli)** Implemented TyperGroup to switch order of commands and option section using help command - ([17407b0](https://github.com/Rakam-AI/rakam_systems/commit/17407b0ad9f5904cc7a994a6eb4908f2f87c6637)) - [somebodyawesome-dev](https://github.com/somebodyawesome-dev)
- **(dependencies)** Remove duplicate pytest-cov entry from dev dependencies - ([78a5f6c](https://github.com/Rakam-AI/rakam_systems/commit/78a5f6c2e2f67965cdbcc578075770377c11fcc0)) - [somebodyawesome-dev](https://github.com/somebodyawesome-dev)
- **(dependencies)** update rakam-systems-tools version to >=0.2.0 across all components - ([8989ebb](https://github.com/Rakam-AI/rakam_systems/commit/8989ebb0d5d41a3ba39bd64ce37465a75cf6c7c3)) - [somebodyawesome-dev](https://github.com/somebodyawesome-dev)
- **(cli)** remove unused variable 'executed_any' in run function - ([b8e8862](https://github.com/Rakam-AI/rakam_systems/commit/b8e8862c3cbaa18865251a5ee34b14338ce2707d)) - [somebodyawesome-dev](https://github.com/somebodyawesome-dev)
- **(agent,vectorestore)** Refactor loaders to use public methods for configuration and validation - ([53d8cab](https://github.com/Rakam-AI/rakam_systems/commit/53d8cab11a62cd059b52386fb33ce20a8284d8e6)) - [somebodyawesome-dev](https://github.com/somebodyawesome-dev)
- **(ci)** update tests scope to include the whole app for each package - ([2995dac](https://github.com/Rakam-AI/rakam_systems/commit/2995dacc9363c88250231f7baa96b73f63c714b6)) - [somebodyawesome-dev](https://github.com/somebodyawesome-dev)
- **(ci, docs)** removed unsupported python version mentions - ([86df0b4](https://github.com/Rakam-AI/rakam_systems/commit/86df0b471a713f303569d91fd86d17e1ee3682db)) - [somebodyawesome-dev](https://github.com/somebodyawesome-dev)
- **(cli)** bumped miinimum python version to 3.10 - ([e16b699](https://github.com/Rakam-AI/rakam_systems/commit/e16b699386d50bfe439e17a936ce4b66b00004a6)) - [somebodyawesome-dev](https://github.com/somebodyawesome-dev)
- **(cli)** regenerated uv lock file - ([358a112](https://github.com/Rakam-AI/rakam_systems/commit/358a112b5108f9010ff199403c09076df14206db)) - [somebodyawesome-dev](https://github.com/somebodyawesome-dev)
- **(core)** bumbed minmum python version to 3.10 - ([6603438](https://github.com/Rakam-AI/rakam_systems/commit/660343825ea80e9808dfbda8cb784d485b0027fd)) - [somebodyawesome-dev](https://github.com/somebodyawesome-dev)
- **(docs)** sync all readme files with all new updates - ([e0a2408](https://github.com/Rakam-AI/rakam_systems/commit/e0a2408bc5e5b4e445796bed7fe060978cefd7bf)) - [somebodyawesome-dev](https://github.com/somebodyawesome-dev)
- **(release)** update publish commands to use testpypi and add rakam-systems build step - ([2ae0109](https://github.com/Rakam-AI/rakam_systems/commit/2ae0109b7312181eabcf42bcf79a88f43d947f9d)) - [somebodyawesome-dev](https://github.com/somebodyawesome-dev)
- **(release)** remove extra spaces in uv publish commands in release.yaml - ([3ae19d5](https://github.com/Rakam-AI/rakam_systems/commit/3ae19d5898e67d8f8c6dcad21c0138e9d1df1cd3)) - [somebodyawesome-dev](https://github.com/somebodyawesome-dev)
- **(review feedback)** Refactor code structure for improved maintainability - ([4b5f12b](https://github.com/Rakam-AI/rakam_systems/commit/4b5f12b737edc41063c1320c693af2bbf5f94944)) - [somebodyawesome-dev](https://github.com/somebodyawesome-dev)
- **(tools)** bumped minmum python version to 3.10 - ([95beb92](https://github.com/Rakam-AI/rakam_systems/commit/95beb927ccc6d24b5826f2921a9eb44b69ea1576)) - [somebodyawesome-dev](https://github.com/somebodyawesome-dev)
- (version): bump rakam_systems to 0.3.0 - ([9e1cd04](https://github.com/Rakam-AI/rakam_systems/commit/9e1cd049bae43b7b08d0fc58571512c1b71c1845)) - [somebodyawesome-dev](https://github.com/somebodyawesome-dev)
- added missing blank line - ([370291c](https://github.com/Rakam-AI/rakam_systems/commit/370291cd75dcee65af95e7e5bdae37a73eeace48)) - [Mohamed Bashar Touil](https://github.com/MohamedBasharTouil)
- added asyncio deps in dev group - ([08854e2](https://github.com/Rakam-AI/rakam_systems/commit/08854e2604beaa3fac6e0c86460a242d7444a5a9)) - [somebodyawesome-dev](https://github.com/somebodyawesome-dev)
- bump version to 0.2.4rc18 - ([95b12c2](https://github.com/Rakam-AI/rakam_systems/commit/95b12c2d99ea77bac46cf065a7a4027c69f1397c)) - [somebodyawesome-dev](https://github.com/somebodyawesome-dev)

### Refactoring

- **(loader)** update access to chunker methods in CodeLoader, EmlLoader, and MdLoader classes - ([ff743cb](https://github.com/Rakam-AI/rakam_systems/commit/ff743cbe34d9cf62bcaa83163260ecbfd7a37b54)) - [somebodyawesome-dev](https://github.com/somebodyawesome-dev)
- **(loader)** update access to chunker methods in CodeLoader, EmlLoader, and MdLoader... - ([8a6add0](https://github.com/Rakam-AI/rakam_systems/commit/8a6add06b34175bae9fa9b7c6c2ba2a9780ba5d1)) - [somebodyawesome-dev](https://github.com/somebodyawesome-dev)
- **(vectorstore)** Remove print statements from predict_embeddings method for cleaner logging - ([9b221ce](https://github.com/Rakam-AI/rakam_systems/commit/9b221cef49e1907f892df23ea33d360b0420c108)) - [somebodyawesome-dev](https://github.com/somebodyawesome-dev)

### Tests

- **(tests)** added unittests for code_loader, eml_loader, tabular_loader - ([adfc2d7](https://github.com/Rakam-AI/rakam_systems/commit/adfc2d7031f577f24e196e4d875ebf4e925ebbe0)) - [somebodyawesome-dev](https://github.com/somebodyawesome-dev)
- **(tests,cli)** added unittests for module loader - ([366135a](https://github.com/Rakam-AI/rakam_systems/commit/366135ac922eed8fd73956de17d847003761775f)) - [somebodyawesome-dev](https://github.com/somebodyawesome-dev)
- **(tests,core)** added unittests for config_loader and config_schema - ([854fd1b](https://github.com/Rakam-AI/rakam_systems/commit/854fd1b2a18637763b20a4471f3a534cd9105d0d)) - [somebodyawesome-dev](https://github.com/somebodyawesome-dev)
- **(agen)** added chat_history and example_toools tests - ([962d17d](https://github.com/Rakam-AI/rakam_systems/commit/962d17d7712d3a01d598b25868a108bd9a72e967)) - [somebodyawesome-dev](https://github.com/somebodyawesome-dev)
- **(agent)** add message history tests for JSON and SQL chat history components - ([0610978](https://github.com/Rakam-AI/rakam_systems/commit/061097847709cfdbdf0d6bbc30271813b6ebb49c)) - [somebodyawesome-dev](https://github.com/somebodyawesome-dev)
- **(core)** added tool registery and vs_core unittests - ([fad8565](https://github.com/Rakam-AI/rakam_systems/commit/fad8565790dd57dae954ca8a2f43068637d4298c)) - [somebodyawesome-dev](https://github.com/somebodyawesome-dev)
- **(core, agent)** Add asyncio marker to async test functions and update output_dir handling in TrackingManager tests - ([388ad9e](https://github.com/Rakam-AI/rakam_systems/commit/388ad9e0be7b7fb88e1e101b781a8e04b76a2077)) - [somebodyawesome-dev](https://github.com/somebodyawesome-dev)
- **(loader)** update unittests to chunker methods for CodeLoader, EmlLoader, and MdLoader classes - ([3521116](https://github.com/Rakam-AI/rakam_systems/commit/35211167cd6e1ee454683f7b68e0637869f3dd7c)) - [somebodyawesome-dev](https://github.com/somebodyawesome-dev)
- **(loader)** update unittests to chunker methods - ([a529ece](https://github.com/Rakam-AI/rakam_systems/commit/a529ecebf9032feb5079b4636b467c7b9e7ac1a2)) - [somebodyawesome-dev](https://github.com/somebodyawesome-dev)
- **(tools)** added eval_schema and utils unittests - ([ea31d4e](https://github.com/Rakam-AI/rakam_systems/commit/ea31d4ef60f1a097551155a1840c42192cdc7dde)) - [somebodyawesome-dev](https://github.com/somebodyawesome-dev)
- **(vectorstore)** added md_loader and vs_config unittests - ([c163fb6](https://github.com/Rakam-AI/rakam_systems/commit/c163fb60a1e24d50a2d7812a383a043fc99e4f1e)) - [somebodyawesome-dev](https://github.com/somebodyawesome-dev)
- **(core)** added unittests for several parts of core module - ([31d5174](https://github.com/Rakam-AI/rakam_systems/commit/31d51742c21a677bdc3d459f82a910d3419367f7)) - [somebodyawesome-dev](https://github.com/somebodyawesome-dev)
- **(agent)** added unittests for agent module - ([053414e](https://github.com/Rakam-AI/rakam_systems/commit/053414e4f4052bae13fc98ec14ac0e21d89cba42)) - [somebodyawesome-dev](https://github.com/somebodyawesome-dev)
- **(agent)** added unittests llm_gatewaytests for agent module - ([24b3d5c](https://github.com/Rakam-AI/rakam_systems/commit/24b3d5c61fb69e8009931d369e9341d965103386)) - [somebodyawesome-dev](https://github.com/somebodyawesome-dev)
- **(vectorstore)** add tests for vectorstore module - ([e5b0f93](https://github.com/Rakam-AI/rakam_systems/commit/e5b0f93f5f68401ce8e0762e4daebc2a62086670)) - [somebodyawesome-dev](https://github.com/somebodyawesome-dev)
- **(vectorstore)** added faiss vectorstore tests - ([174e903](https://github.com/Rakam-AI/rakam_systems/commit/174e90351e11659ea3786944bbdadebb78ed3803)) - [somebodyawesome-dev](https://github.com/somebodyawesome-dev)

---

## [0.3.0]

### Bug Fixes

- fix cli's package name - ([132bfac](https://github.com/Rakam-AI/rakam_systems/commit/132bfac75906be73d51e86a1cd1fe196d4cb07ec)) - somebodyawesome-dev
- tests for cli - ([d471b1e](https://github.com/Rakam-AI/rakam_systems/commit/d471b1e3f4ffc971f9446ef742038b046d93f2d3)) - somebodyawesome-dev
- fix deps issues when installing rakam-systems - ([1bcd359](https://github.com/Rakam-AI/rakam_systems/commit/1bcd359eaa9543cb041bea7fbf20c333d01c4cef)) - somebodyawesome-dev
- update broken links between ddocs for ci in docs repo - ([c904842](https://github.com/Rakam-AI/rakam_systems/commit/c904842fb53a1c0ceb6f860a515d52cc2540c68a)) - somebodyawesome-dev
- increase min python version for agent and vs - ([f1f3ca2](https://github.com/Rakam-AI/rakam_systems/commit/f1f3ca23564ebfda956fa71de9c86069dd89a0dc)) - somebodyawesome-dev
- fix core added yaml missing deps - ([ffb3ee5](https://github.com/Rakam-AI/rakam_systems/commit/ffb3ee5658ed80070b06af919e3461a1bd1c22fd)) - somebodyawesome-dev
- added tests setup/config to core - ([f099f2c](https://github.com/Rakam-AI/rakam_systems/commit/f099f2cf14e52bc585050f652bb817aceb4324c9)) - somebodyawesome-dev
- fix importation issue accross all packages - ([3f23383](https://github.com/Rakam-AI/rakam_systems/commit/3f233837cf5acfdff22411d09f6526fcf7c1ce25)) - somebodyawesome-dev
- fix importations issues and bump versions - ([f59e9d5](https://github.com/Rakam-AI/rakam_systems/commit/f59e9d51239813ef079f38ce98be2b14c245ff8f)) - somebodyawesome-dev
- fix typo in docs - ([df284af](https://github.com/Rakam-AI/rakam_systems/commit/df284afafb8869b9d47ab2082d7aedb68a7fa485)) - somebodyawesome-dev
- regenerate quick_start.md - ([47f2867](https://github.com/Rakam-AI/rakam_systems/commit/47f28673737a55040fb3526cc65d85ad47f43a48)) - somebodyawesome-dev
- fix build links for docs - ([f89920a](https://github.com/Rakam-AI/rakam_systems/commit/f89920a0828be4ac617281367f745d9a23efb399)) - somebodyawesome-dev
- allow importation of all modules and object from core - ([31cace2](https://github.com/Rakam-AI/rakam_systems/commit/31cace2008b846a6084f14286d6e090a63d1fce1)) - somebodyawesome-dev
- fixed tests after rakam_eval cli renaming - ([52cc1e4](https://github.com/Rakam-AI/rakam_systems/commit/52cc1e429ce396b63272b78cdc137eaf18230e29)) - somebodyawesome-dev

### Documentation

- Update Python version requirement to 3.10+ - ([00eae81](https://github.com/Rakam-AI/rakam_systems/commit/00eae81c58cd79f971b5c854f7be2709ec4dafe5)) - Yann Rapaport
- rename docs/intro.md to docs/introduction.md (MINOR-03) - ([78163b5](https://github.com/Rakam-AI/rakam_systems/commit/78163b59d3236f0fb8ef99e8dcba75a95348d85d)) - Yann Rapaport
- Update Introduction with vision-aligned content - ([807f293](https://github.com/Rakam-AI/rakam_systems/commit/807f2938631c7a5a326bc40c9e5c6ac1c9ad4e19)) - Yann Rapaport
- rewrite Getting Started Guide for public release - ([75d8108](https://github.com/Rakam-AI/rakam_systems/commit/75d8108eb5aa5bd81f52595b51da766cf7a64941)) - Yann Rapaport
- clean up getting started guide (QSG-MINOR-01, 05, 06, 07) - ([3bea893](https://github.com/Rakam-AI/rakam_systems/commit/3bea8935c2a6dad073844fac1b86cc60da5b0074)) - Yann Rapaport
- add missing docs - ([42e574b](https://github.com/Rakam-AI/rakam_systems/commit/42e574b674c7a973e097174a07d74df5ec2540c3)) - somebodyawesome-dev
- quick update to docs - ([9675ccc](https://github.com/Rakam-AI/rakam_systems/commit/9675ccc1d8309b76f55bb3dc181d89abbd131990)) - somebodyawesome-dev
- specify evalframeworks env are required to use evaluation service - ([5e9a88b](https://github.com/Rakam-AI/rakam_systems/commit/5e9a88bd02e534ee3a6af0a1d4061efe11786676)) - somebodyawesome-dev
- update docs to point to latest rc versions - ([124ca74](https://github.com/Rakam-AI/rakam_systems/commit/124ca7430d77bb51da47cdf72bbe7a420c69a6b3)) - somebodyawesome-dev
- removed duplicate links - ([da8aa20](https://github.com/Rakam-AI/rakam_systems/commit/da8aa20453fda762c3f251dc21dffa5641ae5e4c)) - somebodyawesome-dev
- remove any rc or uncessary version mentions - ([5e2c99d](https://github.com/Rakam-AI/rakam_systems/commit/5e2c99d86d253343b2b2063def0a9dd5cd3577b0)) - somebodyawesome-dev
- removed empty help section in user-guide - ([86faf27](https://github.com/Rakam-AI/rakam_systems/commit/86faf27230b0a0cd511fcc5c653d641898912ba4)) - somebodyawesome-dev
- remove versions specification in docs - ([c7f4f43](https://github.com/Rakam-AI/rakam_systems/commit/c7f4f43ad578cd0b1b780fef038bb352470942cd)) - somebodyawesome-dev
- added initial changelog - ([4400212](https://github.com/Rakam-AI/rakam_systems/commit/440021235ed194046e423434de9b9a1dafbddd82)) - somebodyawesome-dev

### Miscellaneous Chores

- clean up unused modules - ([332b532](https://github.com/Rakam-AI/rakam_systems/commit/332b532bf58c5f5af4026f888c8861a8b8b5b904)) - somebodyawesome-dev
- upgraded vectorstore version and published it - ([d04a7eb](https://github.com/Rakam-AI/rakam_systems/commit/d04a7eb7ad94f9a6a612933adf750ddc58c57d38)) - somebodyawesome-dev
- bump version to latest rc - ([38ddb4c](https://github.com/Rakam-AI/rakam_systems/commit/38ddb4c36695e0867a1ce39d3d0cf1aeeef66fd9)) - somebodyawesome-dev
- added getting started docs - ([02d9ac0](https://github.com/Rakam-AI/rakam_systems/commit/02d9ac088aec370bd53d54f91bf7e26a066fda48)) - somebodyawesome-dev
- bump version for all rakam's packages - ([8abea67](https://github.com/Rakam-AI/rakam_systems/commit/8abea67fae8aecf842a2df784e64229e5d473dd4)) - somebodyawesome-dev
- update main ci branch - ([9b11153](https://github.com/Rakam-AI/rakam_systems/commit/9b11153479f51f9997b0afbf7bb9d7d879993ca8)) - somebodyawesome-dev
- remove editing argument in installation command in docs - ([71eaa47](https://github.com/Rakam-AI/rakam_systems/commit/71eaa47e40fed355738da5d1dc3e027b4cc667e1)) - somebodyawesome-dev
- update versions in docs - ([f422e76](https://github.com/Rakam-AI/rakam_systems/commit/f422e765acf2e7548a5b47e4075f3782f6edd19a)) - somebodyawesome-dev
- fix tools version in docs - ([5f74071](https://github.com/Rakam-AI/rakam_systems/commit/5f740719a00cb20519245e497f81ca13e5e67d7e)) - somebodyawesome-dev
- update docs versions - ([3628d45](https://github.com/Rakam-AI/rakam_systems/commit/3628d455c8c47b9ca2a8a7febefd11d8eb9d22b0)) - somebodyawesome-dev
- update package versions - ([7323e7c](https://github.com/Rakam-AI/rakam_systems/commit/7323e7c070bc9da53cebdf5af236458aabc99983)) - somebodyawesome-dev

### Ft

- added init isssue and pr template - ([dbf5208](https://github.com/Rakam-AI/rakam_systems/commit/dbf52086d2f0d69aa3f4bf38263e90f9f0890c5f)) - somebodyawesome-dev
- added rakam_systems_Core package - ([c1d5cee](https://github.com/Rakam-AI/rakam_systems/commit/c1d5cee50d567b7187bd2b2e1ccc8fc44991b158)) - somebodyawesome-dev
- added docs from rakam-systems-docs repo - ([66a2a8e](https://github.com/Rakam-AI/rakam_systems/commit/66a2a8e2afd980f0aa6605b9960fceb8d12ab1f3)) - somebodyawesome-dev
- added rakam_systems_agents package - ([55d3812](https://github.com/Rakam-AI/rakam_systems/commit/55d381231ec18ace7de3919d728a34c973c7b1d6)) - somebodyawesome-dev
- added rakam_systems_vectorstore - ([f3b2164](https://github.com/Rakam-AI/rakam_systems/commit/f3b2164d2b6994166c0faf904e4fc462940c56c6)) - somebodyawesome-dev
- added rakam_systems_tool - ([4f8c44f](https://github.com/Rakam-AI/rakam_systems/commit/4f8c44f965672a23016fde0c8871f968ba9feb14)) - somebodyawesome-dev
- added rakam_systems_cli package - ([3c378f4](https://github.com/Rakam-AI/rakam_systems/commit/3c378f409e2fad5fdd2c5f567aec8ae919e26b89)) - somebodyawesome-dev
- removed ai_utils modules and refactored ai_core module - ([654151a](https://github.com/Rakam-AI/rakam_systems/commit/654151aa458c1f1d2c33f1cd4d8e595e647590e6)) - somebodyawesome-dev
- added ai_utils module in core to tools - ([ff99601](https://github.com/Rakam-AI/rakam_systems/commit/ff99601d7300a00b5ee12013c3b79f9620e07c91)) - somebodyawesome-dev
- rename rakam_eval cli to "rakam eval" - ([3860cc2](https://github.com/Rakam-AI/rakam_systems/commit/3860cc2f29f48fc3715f4e3a5c484237887a2ebd)) - somebodyawesome-dev
- remove deps to rs_tools from rs_core - ([938e5dc](https://github.com/Rakam-AI/rakam_systems/commit/938e5dc3f840be588898ae89d38893628f99db3a)) - somebodyawesome-dev
- added initial ci for building all packages - ([9b0f5d7](https://github.com/Rakam-AI/rakam_systems/commit/9b0f5d7f145c9db59a442906b5fdc817b7b5cf08)) - somebodyawesome-dev
- make --dry-run arg run checks of eval configuration - ([14c107a](https://github.com/Rakam-AI/rakam_systems/commit/14c107abc2658bde7f26e0da148d5f98812c715a)) - somebodyawesome-dev
