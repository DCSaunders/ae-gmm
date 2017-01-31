#!/usr/bin/perl
use strict;

sub anon_fh {
   local *FH;
   return *FH;
}

my $f_name = $ARGV[0];
my @buckets = (11, 21, 31, 41, 51);
open(my $in, "<", $f_name) 
    or die "$f_name not openable";
my $parallel;
my $parallel_line;
my $p_fname;
if(defined $ARGV[1])
{
    $p_fname = $ARGV[1];
    open($parallel, "<", $ARGV[1]) 
        or die "$p_fname not openable";
}

my %handles;
my %parallel_handles;
my $plus;
my $parallel_plus;
foreach $b (@buckets)
{
    $handles{$b} = anon_fh();
    open($handles{$b}, ">>",  "$f_name.$b")
	or die "$f_name.$b not openable";
    if(defined $ARGV[1]){
    $parallel_handles{$b} = anon_fh();
    open($parallel_handles{$b}, ">>",  "$p_fname.$b")
	or die "$p_fname.$b not openable";
    }
    
}
open($plus, ">>", "$f_name.plus")
    or die "$f_name.plus not openable";
if(defined $ARGV[1]){
open($parallel_plus, ">>", "$p_fname.plus")
    or die "$parallel.plus not openable";
}

while (<$in>)
{
    chomp;
    my $unsplit = $_;
    my @line = split(' ', $unsplit);
    if(defined $ARGV[1])
    {
        $parallel_line = <$parallel>;
    }
	
    if (@line < $buckets[-1]) {
	foreach $b (@buckets)
	{
	    if (@line < $b) {
		print {$handles{$b}} "$unsplit\n";
		if(defined $ARGV[1])
		{
		    print {$parallel_handles{$b}} $parallel_line;
		}
		last;
	    }
	}
    }
    else {
	print $plus "$unsplit\n";
	if(defined $ARGV[1]){
	    print $parallel_plus $parallel_line
	}
    }
}

foreach (@buckets){
    close $handles{$_} or die "$handles{$_}: $!";
}
close $plus or die "$plus: $!";

if (defined $ARGV[1]){
foreach (@buckets){
    close $parallel_handles{$_} or die "$parallel_handles{$_}: $!";
}

close $parallel_plus or die "$p_fname.plus: $!";
}
