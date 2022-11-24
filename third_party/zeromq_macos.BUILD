# Description:
#   OpenCV libraries for video/image processing on MacOS

licenses(["notice"])  # BSD license

exports_files(["LICENSE"])

load("@bazel_skylib//lib:paths.bzl", "paths")

# The path to OpenCV is a combination of the path set for "macos_zeromq"
# in the WORKSPACE file and the prefix here.
PREFIX = "opt/zeromq"

cc_library(
    name = "zeromq",
    srcs = glob(
        [
            paths.join(PREFIX, "lib/libzmq.dylib"),
        ],
    ),
    hdrs = glob([paths.join(PREFIX, "include/*.h*")]),
    includes = [paths.join(PREFIX, "include/")],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)
