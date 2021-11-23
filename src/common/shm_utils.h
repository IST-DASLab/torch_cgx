#pragma once

namespace cgx {
namespace common {
namespace utils {
const int MAX_SHM_NAME_LEN = 1024;

int shmOpen(const char *shmname, const int shmsize, void **shmPtr,
            void **devShmPtr, int create);

int shmUnlink(const char *shmname);

int shmClose(void *shmPtr, void *devShmPtr, const int shmsize);

} // namespace utils
} // namespace common
} // namespace cgx
