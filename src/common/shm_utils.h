/*
 * pytorch-cgx
 *
 * Copyright (C) 2022 Institute of Science and Technology Austria (ISTA).
 * All Rights Reserved.
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

namespace cgx::common::utils {
const int MAX_SHM_NAME_LEN = 1024;

int shmOpen(const char *shmname, const int shmsize, void **shmPtr,
            void **devShmPtr, int create);

int shmUnlink(const char *shmname);

int shmClose(void *shmPtr, void *devShmPtr, const int shmsize);

} // namespace cgx::common::utils
