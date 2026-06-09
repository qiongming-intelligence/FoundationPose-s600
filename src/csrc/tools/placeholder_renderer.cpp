#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

float Clamp01(float value) {
    return std::max(0.0f, std::min(1.0f, value));
}

void SampleTexture(const float* texture, int height, int width, float u, float v, float& r, float& g, float& b) {
    const float uu = Clamp01(u);
    const float vv = Clamp01(v);
    const float x = uu * static_cast<float>(width - 1);
    const float y = vv * static_cast<float>(height - 1);
    const int x0 = std::max(0, std::min(width - 1, static_cast<int>(std::floor(x))));
    const int y0 = std::max(0, std::min(height - 1, static_cast<int>(std::floor(y))));
    const int x1 = std::max(0, std::min(width - 1, x0 + 1));
    const int y1 = std::max(0, std::min(height - 1, y0 + 1));
    const float wx = x - static_cast<float>(x0);
    const float wy = y - static_cast<float>(y0);

    const float* c00 = texture + (y0 * width + x0) * 3;
    const float* c10 = texture + (y0 * width + x1) * 3;
    const float* c01 = texture + (y1 * width + x0) * 3;
    const float* c11 = texture + (y1 * width + x1) * 3;
    for (int c = 0; c < 3; ++c) {
        const float value = (1.0f - wx) * (1.0f - wy) * c00[c] +
                            wx * (1.0f - wy) * c10[c] +
                            (1.0f - wx) * wy * c01[c] +
                            wx * wy * c11[c];
        if (c == 0) {
            r = value;
        } else if (c == 1) {
            g = value;
        } else {
            b = value;
        }
    }
}

}  // namespace

extern "C" int foundationpose_depth_to_xyz_map(
    const float* depth,
    int height,
    int width,
    float fx,
    float fy,
    float cx,
    float cy,
    float invalid_threshold,
    float* xyz) {
    if (depth == nullptr || xyz == nullptr || height <= 0 || width <= 0 || fx == 0.0f || fy == 0.0f) {
        return -1;
    }
    const float inv_fx = 1.0f / fx;
    const float inv_fy = 1.0f / fy;
    const std::int64_t pixels = static_cast<std::int64_t>(height) * static_cast<std::int64_t>(width);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (std::int64_t idx = 0; idx < pixels; ++idx) {
        const float z = depth[idx];
        float x = 0.0f;
        float y = 0.0f;
        float z_out = 0.0f;
        if (z >= invalid_threshold) {
            const int row = static_cast<int>(idx / width);
            const int col = static_cast<int>(idx - static_cast<std::int64_t>(row) * width);
            x = (static_cast<float>(col) - cx) * z * inv_fx;
            y = (static_cast<float>(row) - cy) * z * inv_fy;
            z_out = z;
        }
        xyz[idx * 3 + 0] = x;
        xyz[idx * 3 + 1] = y;
        xyz[idx * 3 + 2] = z_out;
    }
    return 0;
}

extern "C" int foundationpose_inline_transform_rgb_xyz(
    float* rgb_a,
    const std::int64_t* rgb_a_strides,
    float* rgb_b,
    const std::int64_t* rgb_b_strides,
    float* xyz_a,
    const std::int64_t* xyz_a_strides,
    float* xyz_b,
    const std::int64_t* xyz_b_strides,
    const float* pose_centers,
    const float* mesh_diameters,
    std::int64_t batch,
    int height,
    int width,
    float invalid_threshold,
    int normalize_xyz) {
    if (rgb_a == nullptr || rgb_a_strides == nullptr || rgb_b == nullptr || rgb_b_strides == nullptr ||
        xyz_a == nullptr || xyz_a_strides == nullptr || xyz_b == nullptr || xyz_b_strides == nullptr ||
        pose_centers == nullptr || mesh_diameters == nullptr || batch <= 0 || height <= 0 || width <= 0) {
        return -1;
    }

    const std::int64_t pixels = batch * static_cast<std::int64_t>(height) * static_cast<std::int64_t>(width);
    const float rgb_scale = 1.0f / 255.0f;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (std::int64_t idx = 0; idx < pixels; ++idx) {
        const std::int64_t hw = static_cast<std::int64_t>(height) * static_cast<std::int64_t>(width);
        const std::int64_t b = idx / hw;
        const std::int64_t rem = idx - b * hw;
        const std::int64_t y = rem / width;
        const std::int64_t x = rem - y * width;

        for (std::int64_t c = 0; c < 3; ++c) {
            rgb_a[b * rgb_a_strides[0] + c * rgb_a_strides[1] + y * rgb_a_strides[2] + x * rgb_a_strides[3]] *= rgb_scale;
            rgb_b[b * rgb_b_strides[0] + c * rgb_b_strides[1] + y * rgb_b_strides[2] + x * rgb_b_strides[3]] *= rgb_scale;
        }

        const float center0 = pose_centers[b * 3 + 0];
        const float center1 = pose_centers[b * 3 + 1];
        const float center2 = pose_centers[b * 3 + 2];
        const float mesh_diameter = std::max(mesh_diameters[b], 1.0e-8f);
        const float scale = normalize_xyz != 0 ? (2.0f / mesh_diameter) : 1.0f;

        const std::int64_t base_a = b * xyz_a_strides[0] + y * xyz_a_strides[2] + x * xyz_a_strides[3];
        const float z_orig_a = xyz_a[base_a + 2 * xyz_a_strides[1]];
        const bool invalid_a = normalize_xyz != 0 && z_orig_a < invalid_threshold;
        float values_a[3] = {
            (xyz_a[base_a + 0 * xyz_a_strides[1]] - center0) * scale,
            (xyz_a[base_a + 1 * xyz_a_strides[1]] - center1) * scale,
            (xyz_a[base_a + 2 * xyz_a_strides[1]] - center2) * scale,
        };
        for (std::int64_t c = 0; c < 3; ++c) {
            if (invalid_a || (normalize_xyz != 0 && std::fabs(values_a[c]) >= 2.0f)) {
                values_a[c] = 0.0f;
            }
            xyz_a[base_a + c * xyz_a_strides[1]] = values_a[c];
        }

        const std::int64_t base_b = b * xyz_b_strides[0] + y * xyz_b_strides[2] + x * xyz_b_strides[3];
        const float z_orig_b = xyz_b[base_b + 2 * xyz_b_strides[1]];
        const bool invalid_b = normalize_xyz != 0 && z_orig_b < invalid_threshold;
        float values_b[3] = {
            (xyz_b[base_b + 0 * xyz_b_strides[1]] - center0) * scale,
            (xyz_b[base_b + 1 * xyz_b_strides[1]] - center1) * scale,
            (xyz_b[base_b + 2 * xyz_b_strides[1]] - center2) * scale,
        };
        for (std::int64_t c = 0; c < 3; ++c) {
            if (invalid_b || (normalize_xyz != 0 && std::fabs(values_b[c]) >= 2.0f)) {
                values_b[c] = 0.0f;
            }
            xyz_b[base_b + c * xyz_b_strides[1]] = values_b[c];
        }
    }
    return 0;
}

extern "C" int foundationpose_warp_perspective_hwc_batch(
    const float* src,
    int src_height,
    int src_width,
    int channels,
    const float* matrices,
    std::int64_t batch,
    int out_height,
    int out_width,
    int mode,
    float* out_nchw) {
    if (src == nullptr || matrices == nullptr || out_nchw == nullptr || src_height <= 0 || src_width <= 0 ||
        channels <= 0 || batch < 0 || out_height <= 0 || out_width <= 0) {
        return -1;
    }
    if (mode != 0 && mode != 1) {
        return -2;
    }

    const std::int64_t out_pixels = static_cast<std::int64_t>(out_height) * static_cast<std::int64_t>(out_width);
    const std::int64_t total = batch * out_pixels;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (std::int64_t idx = 0; idx < total; ++idx) {
        const std::int64_t b = idx / out_pixels;
        const std::int64_t rem = idx - b * out_pixels;
        const int y = static_cast<int>(rem / out_width);
        const int x = static_cast<int>(rem - static_cast<std::int64_t>(y) * out_width);
        const float* m = matrices + b * 9;
        const float xf = static_cast<float>(x);
        const float yf = static_cast<float>(y);
        const float denom = m[6] * xf + m[7] * yf + m[8];

        float sx = 0.0f;
        float sy = 0.0f;
        bool valid_coord = std::fabs(denom) > 1.0e-8f;
        if (valid_coord) {
            sx = (m[0] * xf + m[1] * yf + m[2]) / denom;
            sy = (m[3] * xf + m[4] * yf + m[5]) / denom;
        }

        for (int c = 0; c < channels; ++c) {
            float value = 0.0f;
            if (valid_coord && mode == 0) {
                const int ix = static_cast<int>(std::floor(sx + 0.5f));
                const int iy = static_cast<int>(std::floor(sy + 0.5f));
                if (ix >= 0 && ix < src_width && iy >= 0 && iy < src_height) {
                    value = src[(static_cast<std::int64_t>(iy) * src_width + ix) * channels + c];
                }
            } else if (valid_coord) {
                const int x0 = static_cast<int>(std::floor(sx));
                const int y0 = static_cast<int>(std::floor(sy));
                const int x1 = x0 + 1;
                const int y1 = y0 + 1;
                const float wx = sx - static_cast<float>(x0);
                const float wy = sy - static_cast<float>(y0);
                if (x0 >= 0 && x0 < src_width && y0 >= 0 && y0 < src_height) {
                    value += (1.0f - wx) * (1.0f - wy) * src[(static_cast<std::int64_t>(y0) * src_width + x0) * channels + c];
                }
                if (x1 >= 0 && x1 < src_width && y0 >= 0 && y0 < src_height) {
                    value += wx * (1.0f - wy) * src[(static_cast<std::int64_t>(y0) * src_width + x1) * channels + c];
                }
                if (x0 >= 0 && x0 < src_width && y1 >= 0 && y1 < src_height) {
                    value += (1.0f - wx) * wy * src[(static_cast<std::int64_t>(y1) * src_width + x0) * channels + c];
                }
                if (x1 >= 0 && x1 < src_width && y1 >= 0 && y1 < src_height) {
                    value += wx * wy * src[(static_cast<std::int64_t>(y1) * src_width + x1) * channels + c];
                }
            }
            out_nchw[(b * channels + c) * out_pixels + rem] = value;
        }
    }
    return 0;
}

extern "C" int foundationpose_placeholder_render_triangles(
    const float* x,
    const float* y,
    const float* z,
    const std::uint8_t* valid,
    std::int64_t vertex_count,
    const float* cam_xyz,
    const float* colors,
    const float* normals_cam,
    const std::int64_t* faces,
    std::int64_t face_count,
    int height,
    int width,
    int use_light,
    const float* light_dir_neg,
    float w_ambient,
    float w_diffuse,
    int use_texture,
    const float* texture,
    int texture_height,
    int texture_width,
    const float* uv,
    std::int64_t uv_count,
    const std::int64_t* uv_faces,
    float* depth,
    float* xyz,
    float* color) {
    if (x == nullptr || y == nullptr || z == nullptr || valid == nullptr || cam_xyz == nullptr ||
        colors == nullptr || faces == nullptr || depth == nullptr || xyz == nullptr || color == nullptr ||
        height <= 0 || width <= 0 || vertex_count <= 0 || face_count < 0) {
        return -1;
    }
    if (use_light != 0 && (normals_cam == nullptr || light_dir_neg == nullptr)) {
        return -2;
    }
    if (use_texture != 0 && (texture == nullptr || texture_height <= 0 || texture_width <= 0 ||
                             uv == nullptr || uv_faces == nullptr || uv_count <= 0)) {
        return -3;
    }

    const float inside_eps = -1.0e-4f;
    const float z_eps = 1.0e-4f;

    for (std::int64_t face_i = 0; face_i < face_count; ++face_i) {
        const std::int64_t f0 = faces[face_i * 3 + 0];
        const std::int64_t f1 = faces[face_i * 3 + 1];
        const std::int64_t f2 = faces[face_i * 3 + 2];
        if (f0 < 0 || f1 < 0 || f2 < 0 || f0 >= vertex_count || f1 >= vertex_count || f2 >= vertex_count) {
            continue;
        }
        if (valid[f0] == 0 || valid[f1] == 0 || valid[f2] == 0) {
            continue;
        }

        std::int64_t tu0 = 0;
        std::int64_t tu1 = 0;
        std::int64_t tu2 = 0;
        if (use_texture != 0) {
            tu0 = uv_faces[face_i * 3 + 0];
            tu1 = uv_faces[face_i * 3 + 1];
            tu2 = uv_faces[face_i * 3 + 2];
            if (tu0 < 0 || tu1 < 0 || tu2 < 0 || tu0 >= uv_count || tu1 >= uv_count || tu2 >= uv_count) {
                continue;
            }
        }

        const float xs0 = x[f0];
        const float xs1 = x[f1];
        const float xs2 = x[f2];
        const float ys0 = y[f0];
        const float ys1 = y[f1];
        const float ys2 = y[f2];

        int xmin = static_cast<int>(std::floor(std::min(xs0, std::min(xs1, xs2))));
        int xmax = static_cast<int>(std::ceil(std::max(xs0, std::max(xs1, xs2))));
        int ymin = static_cast<int>(std::floor(std::min(ys0, std::min(ys1, ys2))));
        int ymax = static_cast<int>(std::ceil(std::max(ys0, std::max(ys1, ys2))));
        xmin = std::max(0, xmin);
        ymin = std::max(0, ymin);
        xmax = std::min(width - 1, xmax);
        ymax = std::min(height - 1, ymax);
        if (xmax < xmin || ymax < ymin) {
            continue;
        }

        const float area = (xs1 - xs0) * (ys2 - ys0) - (ys1 - ys0) * (xs2 - xs0);
        if (std::fabs(area) < 1.0e-6f) {
            continue;
        }

        for (int py = ymin; py <= ymax; ++py) {
            const float gy = static_cast<float>(py) + 0.5f;
            for (int px = xmin; px <= xmax; ++px) {
                const float gx = static_cast<float>(px) + 0.5f;
                const float w0 = ((xs1 - gx) * (ys2 - gy) - (ys1 - gy) * (xs2 - gx)) / area;
                const float w1 = ((xs2 - gx) * (ys0 - gy) - (ys2 - gy) * (xs0 - gx)) / area;
                const float w2 = 1.0f - w0 - w1;
                if (w0 < inside_eps || w1 < inside_eps || w2 < inside_eps) {
                    continue;
                }

                const float zz = w0 * z[f0] + w1 * z[f1] + w2 * z[f2];
                if (zz <= z_eps) {
                    continue;
                }
                const int pix = py * width + px;
                if (depth[pix] != 0.0f && zz >= depth[pix]) {
                    continue;
                }

                depth[pix] = zz;
                for (int c = 0; c < 3; ++c) {
                    xyz[pix * 3 + c] = w0 * cam_xyz[f0 * 3 + c] + w1 * cam_xyz[f1 * 3 + c] + w2 * cam_xyz[f2 * 3 + c];
                }

                float r = 0.0f;
                float g = 0.0f;
                float b = 0.0f;
                if (use_texture != 0) {
                    const float u = w0 * uv[tu0 * 2 + 0] + w1 * uv[tu1 * 2 + 0] + w2 * uv[tu2 * 2 + 0];
                    const float v = w0 * uv[tu0 * 2 + 1] + w1 * uv[tu1 * 2 + 1] + w2 * uv[tu2 * 2 + 1];
                    SampleTexture(texture, texture_height, texture_width, u, v, r, g, b);
                } else {
                    r = w0 * colors[f0 * 3 + 0] + w1 * colors[f1 * 3 + 0] + w2 * colors[f2 * 3 + 0];
                    g = w0 * colors[f0 * 3 + 1] + w1 * colors[f1 * 3 + 1] + w2 * colors[f2 * 3 + 1];
                    b = w0 * colors[f0 * 3 + 2] + w1 * colors[f1 * 3 + 2] + w2 * colors[f2 * 3 + 2];
                }

                if (use_light != 0) {
                    float nx = w0 * normals_cam[f0 * 3 + 0] + w1 * normals_cam[f1 * 3 + 0] + w2 * normals_cam[f2 * 3 + 0];
                    float ny = w0 * normals_cam[f0 * 3 + 1] + w1 * normals_cam[f1 * 3 + 1] + w2 * normals_cam[f2 * 3 + 1];
                    float nz = w0 * normals_cam[f0 * 3 + 2] + w1 * normals_cam[f1 * 3 + 2] + w2 * normals_cam[f2 * 3 + 2];
                    const float norm = std::sqrt(nx * nx + ny * ny + nz * nz);
                    if (norm > 1.0e-8f) {
                        nx /= norm;
                        ny /= norm;
                        nz /= norm;
                    }
                    const float diffuse = Clamp01(nx * light_dir_neg[0] + ny * light_dir_neg[1] + nz * light_dir_neg[2]);
                    const float scale = w_ambient + diffuse * w_diffuse;
                    r = Clamp01(r * scale);
                    g = Clamp01(g * scale);
                    b = Clamp01(b * scale);
                }

                color[pix * 3 + 0] = r;
                color[pix * 3 + 1] = g;
                color[pix * 3 + 2] = b;
            }
        }
    }

    return 0;
}

extern "C" int foundationpose_placeholder_render_points(
    const float* x,
    const float* y,
    const float* z,
    const std::uint8_t* valid,
    std::int64_t vertex_count,
    const float* cam_xyz,
    const float* colors,
    const std::int64_t* point_indices,
    std::int64_t point_count,
    int height,
    int width,
    float* depth,
    float* xyz,
    float* color) {
    if (x == nullptr || y == nullptr || z == nullptr || valid == nullptr || cam_xyz == nullptr ||
        colors == nullptr || point_indices == nullptr || depth == nullptr || xyz == nullptr || color == nullptr ||
        height <= 0 || width <= 0 || vertex_count <= 0 || point_count < 0) {
        return -1;
    }

    for (std::int64_t i = 0; i < point_count; ++i) {
        const std::int64_t idx = point_indices[i];
        if (idx < 0 || idx >= vertex_count || valid[idx] == 0) {
            continue;
        }
        const int px = static_cast<int>(std::floor(x[idx] + 0.5f));
        const int py = static_cast<int>(std::floor(y[idx] + 0.5f));
        if (px < 0 || px >= width || py < 0 || py >= height) {
            continue;
        }
        const float zz = z[idx];
        if (zz <= 1.0e-4f) {
            continue;
        }
        const int pix = py * width + px;
        if (depth[pix] != 0.0f && zz >= depth[pix]) {
            continue;
        }
        depth[pix] = zz;
        xyz[pix * 3 + 0] = cam_xyz[idx * 3 + 0];
        xyz[pix * 3 + 1] = cam_xyz[idx * 3 + 1];
        xyz[pix * 3 + 2] = cam_xyz[idx * 3 + 2];
        color[pix * 3 + 0] = colors[idx * 3 + 0];
        color[pix * 3 + 1] = colors[idx * 3 + 1];
        color[pix * 3 + 2] = colors[idx * 3 + 2];
    }

    return 0;
}

extern "C" int foundationpose_placeholder_render_pose(
    const float* pos,
    std::int64_t vertex_count,
    const float* pose,
    float fx,
    float fy,
    float cx,
    float cy,
    float full_height,
    float full_width,
    const float* bbox,
    int use_bbox,
    const float* colors,
    const float* normals,
    const std::int64_t* faces,
    std::int64_t face_count,
    int height,
    int width,
    int use_light,
    const float* light_dir_neg,
    float w_ambient,
    float w_diffuse,
    int use_texture,
    const float* texture,
    int texture_height,
    int texture_width,
    const float* uv,
    std::int64_t uv_count,
    const std::int64_t* uv_faces,
    const std::int64_t* point_indices,
    std::int64_t point_count,
    float* depth,
    float* xyz,
    float* color) {
    if (pos == nullptr || pose == nullptr || colors == nullptr || faces == nullptr ||
        point_indices == nullptr || depth == nullptr || xyz == nullptr || color == nullptr ||
        vertex_count <= 0 || height <= 0 || width <= 0) {
        return -1;
    }
    if (use_bbox != 0 && bbox == nullptr) {
        return -2;
    }
    if (use_light != 0 && (normals == nullptr || light_dir_neg == nullptr)) {
        return -3;
    }

    std::vector<float> x(static_cast<std::size_t>(vertex_count));
    std::vector<float> y(static_cast<std::size_t>(vertex_count));
    std::vector<float> z(static_cast<std::size_t>(vertex_count));
    std::vector<std::uint8_t> valid(static_cast<std::size_t>(vertex_count));
    std::vector<float> cam_xyz(static_cast<std::size_t>(vertex_count) * 3U);
    std::vector<float> normals_cam;
    if (use_light != 0) {
        normals_cam.resize(static_cast<std::size_t>(vertex_count) * 3U);
    }

    float left = 0.0f;
    float top = 0.0f;
    float box_width = 1.0f;
    float box_height = 1.0f;
    if (use_bbox != 0) {
        left = bbox[0];
        top = bbox[1];
        const float right = bbox[2];
        const float bottom = bbox[3];
        box_width = std::max(right - left, 1.0f);
        box_height = std::max(bottom - top, 1.0f);
    }
    const float full_width_denom = std::max(full_width - 1.0f, 1.0f);
    const float full_height_denom = std::max(full_height - 1.0f, 1.0f);
    const float out_width = static_cast<float>(width - 1);
    const float out_height = static_cast<float>(height - 1);

    for (std::int64_t i = 0; i < vertex_count; ++i) {
        const float px = pos[i * 3 + 0];
        const float py = pos[i * 3 + 1];
        const float pz = pos[i * 3 + 2];
        const float cam_x = pose[0] * px + pose[1] * py + pose[2] * pz + pose[3];
        const float cam_y = pose[4] * px + pose[5] * py + pose[6] * pz + pose[7];
        const float cam_z = pose[8] * px + pose[9] * py + pose[10] * pz + pose[11];
        cam_xyz[i * 3 + 0] = cam_x;
        cam_xyz[i * 3 + 1] = cam_y;
        cam_xyz[i * 3 + 2] = cam_z;
        z[static_cast<std::size_t>(i)] = cam_z;
        valid[static_cast<std::size_t>(i)] = cam_z > 1.0e-4f ? 1U : 0U;
        const float z_safe = std::max(cam_z, 1.0e-4f);
        const float u = fx * cam_x / z_safe + cx;
        const float v = fy * cam_y / z_safe + cy;
        if (use_bbox != 0) {
            x[static_cast<std::size_t>(i)] = (u - left) / box_width * out_width;
            y[static_cast<std::size_t>(i)] = (v - top) / box_height * out_height;
        } else {
            x[static_cast<std::size_t>(i)] = u / full_width_denom * out_width;
            y[static_cast<std::size_t>(i)] = v / full_height_denom * out_height;
        }

        if (use_light != 0) {
            const float nx = normals[i * 3 + 0];
            const float ny = normals[i * 3 + 1];
            const float nz = normals[i * 3 + 2];
            normals_cam[i * 3 + 0] = pose[0] * nx + pose[1] * ny + pose[2] * nz;
            normals_cam[i * 3 + 1] = pose[4] * nx + pose[5] * ny + pose[6] * nz;
            normals_cam[i * 3 + 2] = pose[8] * nx + pose[9] * ny + pose[10] * nz;
        }
    }

    const int tri_rc = foundationpose_placeholder_render_triangles(
        x.data(), y.data(), z.data(), valid.data(), vertex_count,
        cam_xyz.data(), colors, use_light != 0 ? normals_cam.data() : nullptr,
        faces, face_count, height, width, use_light, light_dir_neg,
        w_ambient, w_diffuse, use_texture, texture, texture_height, texture_width,
        uv, uv_count, uv_faces, depth, xyz, color);
    if (tri_rc != 0) {
        return tri_rc;
    }

    return foundationpose_placeholder_render_points(
        x.data(), y.data(), z.data(), valid.data(), vertex_count,
        cam_xyz.data(), colors, point_indices, point_count,
        height, width, depth, xyz, color);
}

extern "C" int foundationpose_placeholder_render_pose_batch(
    const float* pos,
    std::int64_t vertex_count,
    const float* poses,
    std::int64_t pose_count,
    float fx,
    float fy,
    float cx,
    float cy,
    float full_height,
    float full_width,
    const float* bboxes,
    int use_bbox,
    const float* colors,
    const float* normals,
    const std::int64_t* faces,
    std::int64_t face_count,
    int height,
    int width,
    int use_light,
    const float* light_dir_neg,
    float w_ambient,
    float w_diffuse,
    int use_texture,
    const float* texture,
    int texture_height,
    int texture_width,
    const float* uv,
    std::int64_t uv_count,
    const std::int64_t* uv_faces,
    const std::int64_t* point_indices,
    std::int64_t point_count,
    float* depth,
    float* xyz,
    float* color) {
    if (poses == nullptr || pose_count < 0 || depth == nullptr || xyz == nullptr || color == nullptr ||
        height <= 0 || width <= 0) {
        return -1;
    }
    if (use_bbox != 0 && bboxes == nullptr) {
        return -2;
    }

    const std::int64_t pixels = static_cast<std::int64_t>(height) * static_cast<std::int64_t>(width);
    std::atomic<int> first_error{0};
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (std::int64_t i = 0; i < pose_count; ++i) {
        if (first_error.load(std::memory_order_relaxed) != 0) {
            continue;
        }
        const float* bbox = use_bbox != 0 ? bboxes + i * 4 : nullptr;
        const int rc = foundationpose_placeholder_render_pose(
            pos, vertex_count, poses + i * 16,
            fx, fy, cx, cy, full_height, full_width, bbox, use_bbox,
            colors, normals, faces, face_count, height, width, use_light,
            light_dir_neg, w_ambient, w_diffuse, use_texture, texture,
            texture_height, texture_width, uv, uv_count, uv_faces,
            point_indices, point_count,
            depth + i * pixels,
            xyz + i * pixels * 3,
            color + i * pixels * 3);
        if (rc != 0) {
            int expected = 0;
            first_error.compare_exchange_strong(expected, rc, std::memory_order_relaxed);
        }
    }
    return first_error.load(std::memory_order_relaxed);
}
