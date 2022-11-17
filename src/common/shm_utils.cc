/*************************************************************************
* Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
*
* Modifications copyright (C) 2022, IST Austria.
************************************************************************/

#include "shm_utils.h"
#include <cstring>
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#if HAVE_CUDA
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#elif HAVE_ROCM
#include <hip/hip_runtime.h>
#endif
#include "compression/gpu_common.h"

#define SYS_CHECK(call, name)                                                  \
  do {                                                                         \
    int retval;                                                                \
    SYS_CHECKVAL(call, name, retval);                                          \
  } while (false)

#define SYS_CHECKSYNC(call, name, retval)                                      \
  do {                                                                         \
    retval = call;                                                             \
    if (retval == -1 &&                                                        \
        (errno == EINTR || errno == EWOULDBLOCK || errno == EAGAIN)) {         \
      printf("Call to " name " returned %s, retrying", strerror(errno));       \
    } else {                                                                   \
      break;                                                                   \
    }                                                                          \
  } while (true)

#define SYS_CHECKVAL(call, name, retval)                                       \
  do {                                                                         \
    SYS_CHECKSYNC(call, name, retval);                                         \
    if (retval == -1) {                                                        \
      printf("Call to " name " failed : %s", strerror(errno));                 \
      return 1;                                                                \
    }                                                                          \
  } while (false)


namespace cgx {
namespace common {
namespace utils {


int shm_allocate(int fd, const int shmsize) {
  int err = posix_fallocate(fd, 0, shmsize);
  if (err) {
    errno = err;
    return -1;
  }
  return 0;
}

int shm_map(int fd, const int shmsize, void **ptr) {
  *ptr = mmap(NULL, shmsize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  return (*ptr == MAP_FAILED) ? -1 : 0;
}

static int shm_setup(const char *shmname, const int shmsize, int *fd, void **ptr,
                    int create) {
  SYS_CHECKVAL(shm_open(shmname, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR),
               "shm_open", *fd);
  if (create)
    SYS_CHECK(shm_allocate(*fd, shmsize), "posix_fallocate");
  SYS_CHECK(shm_map(*fd, shmsize, ptr), "mmap");
  close(*fd);
  *fd = -1;
  if (create)
    memset(*ptr, 0, shmsize);
  return 0;
}

int shmOpen(const char *shmname, const int shmsize, void **shmPtr,
                   void **devShmPtr, int create) {
  int fd = -1;
  void *ptr = MAP_FAILED;
  int res = 0;

  res = shm_setup(shmname, shmsize, &fd, &ptr, create);
  if (res > 0)
    goto sysError;
#if HAVE_CUDA
  if ((res = cudaHostRegister(ptr, shmsize, cudaHostRegisterMapped)) !=
      cudaSuccess ||
      (res = cudaHostGetDevicePointer(devShmPtr, ptr, 0)) != cudaSuccess)
    goto gpuError;
#elif HAVE_ROCM
  if ((res = hipHostRegister(ptr, shmsize, hipHostRegisterMapped)) !=
      hipSuccess ||
      (res = hipHostGetDevicePointer(devShmPtr, ptr, 0)) != hipSuccess)
    goto gpuError;
#endif
  *shmPtr = ptr;
  return 0;
sysError:
  printf("Error while %s shared memory segment %s (size %d)\n",
         create ? "creating" : "attaching to", shmname, shmsize);
gpuError:
  if (fd != -1)
    close(fd);
  if (create)
    shm_unlink(shmname);
  if (ptr != MAP_FAILED)
    munmap(ptr, shmsize);
  *shmPtr = NULL;
  return res;
}

int shmUnlink(const char *shmname) {
  if (shmname != NULL)
    SYS_CHECK(shm_unlink(shmname), "shm_unlink");
  return 0;
}

int shmClose(void *shmPtr, void *devShmPtr, const int shmsize) {
#if HAVE_CUDA
  CUDA_CHECK(cudaHostUnregister(shmPtr));
#elif HAVE_ROCM
  HIP_CHECK(hipHostUnregister(shmPtr));
#endif
  if (munmap(shmPtr, shmsize) != 0) {
    printf("munmap of shared memory failed\n");
    return 1;
  }
  return 0;
}

} // namespace utils
} // namespace common
} // namespace cgx

