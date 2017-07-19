"""
    HDF5Logger

Creates an object for logging frames of data to an HDF5 file. The frames are
expected to have the same size, and the total number of frames to log is
expected to be known in advance. It currently works with scalars, vectors, and
matrices of basic types that the HDF5 package supports.

# Examples
```julia-repl
    log = Log("my_file.h5") # Create a logger.
    num_frames = 100        # Set the total number of frames to log.
    example_data = [1., 2., 3.] # Create an example of a frame of data.
    add!(log, "/a_vector", example_data, num_frames) # Create the stream.
    log!(log, "/a_vector", [4., 5., 6.]) # Log a single frame of data.
    close!(log) # Always clean up when done. Use try-catch to make sure.
```

"""
module HDF5Logger

using HDF5

export Log, add!, log!, close!

mutable struct Stream
    count::Int64
    length::Int64
    rank::Int64
    dataset::HDF5Dataset
end

struct Log
    streams::Dict{String,Stream}
    file_id::HDF5File
    function Log(file_name::String)
        new(Dict{String,Stream}(), h5open(file_name, "w"));
    end
end

function prepare_group!(log::Log, slug::String)
    groups = filter(x->!isempty(x), split(slug)) # Explode to group names
    group_id = g_open(log.file_id, "/") # Start at top group
    for k = 1:length(groups)-1 # For each group up to dataset
        if exists(group_id, groups[k])
            println("Group '$(groups[k]))' already exists.")
            group_id = g_open(group_id, String(groups[k]))
        else
            println("Creating group '$(groups[k]))'.")
            group_id = g_create(group_id, String(groups[k]))
        end
    end
    return (group_id, String(groups[end])) # Group ID and name of dataset
end

function add!(log::Log, group_name::String, data, num_samples::Int64)
    dims = isbits(data) ? 1 : size(data)
    group_id, group_name    = prepare_group!(log, group_name);
    dataset_id              = d_create(group_id, group_name,
                                       datatype(eltype(data)),
                                       dataspace(dims..., num_samples));
    log.streams[group_name] = Stream(0, num_samples, length(dims), dataset_id)
end

function log!(log::Log, slug::String, data)
    @assert(haskey(log.streams, slug),
            "The logger doesn't have that key. Perhaps you need to `add` it?")
    @assert(log.streams[slug].count < log.streams[slug].length,
            "We've already used up all of the allocated space for this stream!")
    log.streams[slug].count += 1;
    index = (repeat([:], outer=log.streams[slug].rank)...,
             log.streams[slug].count); # TODO: Is there a better way to do this?
    log.streams[slug].dataset[index...] = data
end

function close!(log::Log)
    close(log.file_id)
end

end # module
